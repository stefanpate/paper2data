"""LLM-based document summarization with per-document on-disk cache.

Summaries are deterministic for (model_tag, prompt_version, document, target_words,
temperature, seed), so each (doc, fraction) pair is cached individually. This lets
us add new fractions without re-summarizing untouched ones.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Default prompt + version are kept here as a fallback so library callers can
# use this module without wiring through a config. The canonical copy lives in
# `conf/prompts/zero_shot_v1.yaml`; bump the version there (and here) whenever the
# template changes — the version participates in the cache key.
DEFAULT_PROMPT_VERSION = "v1"

DEFAULT_SUMMARY_PROMPT = """You are summarizing a Usenet newsgroup post for a downstream classifier.
Produce a faithful summary in approximately {target_words} words. Preserve topical \
vocabulary (named entities, jargon, technical terms) — these are the strongest \
classification signal. Do not add commentary or preamble.

POST:
{document}

SUMMARY:"""


@dataclass
class SummaryResult:
    text: str
    cache_key: str
    cache_hit: bool
    target_words: int
    source_words: int


def _doc_words(doc: str) -> int:
    return max(1, len(doc.split()))


def _doc_sha1(doc: str) -> str:
    return hashlib.sha1(doc.encode("utf-8", errors="replace")).hexdigest()[:16]


def _cache_key(*, model_tag: str, doc_sha1: str, target_words: int,
               temperature: float, seed: int, prompt_version: str) -> str:
    payload = json.dumps(
        {
            "v": prompt_version,
            "model_tag": model_tag,
            "doc": doc_sha1,
            "target_words": target_words,
            "temperature": temperature,
            "seed": seed,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha1(payload).hexdigest()[:24]


def target_words_for(doc: str, fraction: float) -> int:
    """How many words to ask the LLM to produce, given a fraction of source length."""
    return max(8, math.ceil(fraction * _doc_words(doc)))


def _summary_max_tokens(target_words: int) -> int:
    """Output-token budget for a summary of ~target_words (API backend needs it)."""
    return max(64, int(target_words * 2) + 64)


def _write_summary_cache(
    txt_path: Path, meta_path: Path, text: str, *,
    prompt_version: str, model_tag: str, doc_sha1: str,
    target_words: int, source_words: int, fraction: float,
    temperature: float, seed: int,
) -> None:
    txt_path.write_text(text)
    meta_path.write_text(json.dumps({
        "prompt_version": prompt_version,
        "model_tag": model_tag,
        "doc_sha1": doc_sha1,
        "target_words": target_words,
        "source_words": source_words,
        "fraction": fraction,
        "temperature": temperature,
        "seed": seed,
    }, indent=2))


def summarize_doc(
    doc: str,
    *,
    fraction: float,
    model_tag: str,
    cache_dir: str | Path,
    num_ctx: int = 16384,
    temperature: float = 0.0,
    seed: int = 0,
    client=None,
    prompt_template: str = DEFAULT_SUMMARY_PROMPT,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> SummaryResult:
    """Summarize one document at the given length fraction, with on-disk cache.

    fraction == 1.0 is a sentinel: return the raw doc unmodified, no LLM call,
    no cache write — this lets the downstream classifier evaluate on full docs
    as a baseline using exactly the same code path.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    source_words = _doc_words(doc)

    if fraction >= 1.0:
        return SummaryResult(
            text=doc,
            cache_key="raw",
            cache_hit=True,
            target_words=source_words,
            source_words=source_words,
        )

    target_words = target_words_for(doc, fraction)
    key = _cache_key(
        model_tag=model_tag,
        doc_sha1=_doc_sha1(doc),
        target_words=target_words,
        temperature=temperature,
        seed=seed,
        prompt_version=prompt_version,
    )
    txt_path = cache_dir / f"{key}.txt"
    meta_path = cache_dir / f"{key}.json"

    if txt_path.exists():
        return SummaryResult(
            text=txt_path.read_text(),
            cache_key=key,
            cache_hit=True,
            target_words=target_words,
            source_words=source_words,
        )

    if client is None:
        from paper2data.llm_providers import OllamaProvider
        import ollama
        client = OllamaProvider(ollama.Client(host=os.environ.get("OLLAMA_HOST")))

    prompt = prompt_template.format(target_words=target_words, document=doc)
    resp = client.generate(
        model=model_tag,
        prompt=prompt,
        options={
            "temperature": temperature,
            "num_ctx": num_ctx,
            "seed": seed,
        },
        max_tokens=_summary_max_tokens(target_words),
    )
    text = resp["response"].strip() if isinstance(resp, dict) else resp.response.strip()

    _write_summary_cache(
        txt_path, meta_path, text,
        prompt_version=prompt_version, model_tag=model_tag,
        doc_sha1=_doc_sha1(doc), target_words=target_words,
        source_words=source_words, fraction=fraction,
        temperature=temperature, seed=seed,
    )

    return SummaryResult(
        text=text,
        cache_key=key,
        cache_hit=False,
        target_words=target_words,
        source_words=source_words,
    )


def summarize_corpus(
    docs: list[str],
    *,
    fraction: float,
    model_tag: str,
    cache_dir: str | Path,
    num_ctx: int = 16384,
    temperature: float = 0.0,
    seed: int = 0,
    show_progress: bool = True,
    prompt_template: str = DEFAULT_SUMMARY_PROMPT,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    client=None,
    provider_cfg=None,
) -> list[SummaryResult]:
    """Summarize a list of documents. Cache hits are O(disk read); misses call the LLM.

    `client` is an LLM provider (see paper2data.llm_providers). If omitted, one is
    built from `provider_cfg` (an llm config node), falling back to a local Ollama
    client for backward compatibility. Cache-misses are routed to the provider's
    Batch API when it supports one, otherwise run serially or concurrently
    according to `client.concurrency`.
    """
    if client is None:
        if provider_cfg is not None:
            from paper2data.llm_providers import make_provider
            client = make_provider(provider_cfg)
        else:
            from paper2data.llm_providers import OllamaProvider
            import ollama
            host = os.environ.get("OLLAMA_HOST")
            log.info("summarize_corpus: ollama host=%r", host)
            client = OllamaProvider(ollama.Client(host=host))

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: resolve every doc that needs no LLM call (raw sentinel or cache hit)
    # and collect the rest as misses we still have to generate.
    results: list[SummaryResult | None] = [None] * len(docs)
    misses: list[int] = []
    for i, doc in enumerate(docs):
        source_words = _doc_words(doc)
        if fraction >= 1.0:
            results[i] = SummaryResult(doc, "raw", True, source_words, source_words)
            continue
        target_words = target_words_for(doc, fraction)
        key = _cache_key(
            model_tag=model_tag, doc_sha1=_doc_sha1(doc),
            target_words=target_words, temperature=temperature,
            seed=seed, prompt_version=prompt_version,
        )
        txt_path = cache_dir / f"{key}.txt"
        if txt_path.exists():
            results[i] = SummaryResult(
                txt_path.read_text(), key, True, target_words, source_words
            )
        else:
            misses.append(i)

    t0 = time.perf_counter()
    if misses and client.supports_batch():
        _summarize_batch(
            docs, misses, results,
            fraction=fraction, model_tag=model_tag, cache_dir=cache_dir,
            temperature=temperature, seed=seed,
            prompt_template=prompt_template, prompt_version=prompt_version,
            client=client,
        )
    elif misses:
        _summarize_loop(
            docs, misses, results,
            fraction=fraction, model_tag=model_tag, cache_dir=cache_dir,
            num_ctx=num_ctx, temperature=temperature, seed=seed,
            prompt_template=prompt_template, prompt_version=prompt_version,
            client=client, show_progress=show_progress,
        )
    elapsed = time.perf_counter() - t0

    log.info(
        "Summarized %d docs at fraction=%.2f (%d cache hits, %d misses) in %.1fs",
        len(docs), fraction, len(docs) - len(misses), len(misses), elapsed,
    )
    return [r for r in results if r is not None]


def _summarize_loop(docs, misses, results, *, fraction, model_tag, cache_dir,
                    num_ctx, temperature, seed, prompt_template, prompt_version,
                    client, show_progress) -> None:
    """Run misses through summarize_doc, serially or concurrently per provider."""
    def _one(i: int) -> tuple[int, SummaryResult]:
        return i, summarize_doc(
            docs[i], fraction=fraction, model_tag=model_tag, cache_dir=cache_dir,
            num_ctx=num_ctx, temperature=temperature, seed=seed, client=client,
            prompt_template=prompt_template, prompt_version=prompt_version,
        )

    concurrency = int(getattr(client, "concurrency", 1) or 1)
    iterator = misses
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(misses, desc=f"summarize@{fraction}")
        except ImportError:
            pass

    if concurrency <= 1:
        for i in iterator:
            idx, r = _one(i)
            results[idx] = r
        return

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for idx, r in ex.map(_one, iterator):
            results[idx] = r


def _summarize_batch(docs, misses, results, *, fraction, model_tag, cache_dir,
                     temperature, seed, prompt_template, prompt_version,
                     client) -> None:
    """Submit all misses as one batch, then backfill the per-doc cache.

    Identical documents (20NG has cross-posts/reposts) share a content-derived
    cache key, so they are collapsed into a single batch request — batch
    custom_ids must be unique, and one summary legitimately serves every doc that
    hashes to that key. Each such doc is then backfilled from the shared result.
    """
    from paper2data.llm_providers import BatchItem
    # key -> {"indices": [positions in docs], target_words, source_words, doc, prompt}
    by_key: dict[str, dict] = {}
    for i in misses:
        doc = docs[i]
        target_words = target_words_for(doc, fraction)
        key = _cache_key(
            model_tag=model_tag, doc_sha1=_doc_sha1(doc),
            target_words=target_words, temperature=temperature,
            seed=seed, prompt_version=prompt_version,
        )
        entry = by_key.get(key)
        if entry is None:
            by_key[key] = {
                "indices": [i],
                "target_words": target_words,
                "source_words": _doc_words(doc),
                "doc": doc,
                "prompt": prompt_template.format(target_words=target_words, document=doc),
            }
        else:
            entry["indices"].append(i)

    items = [
        BatchItem(custom_id=key, prompt=e["prompt"],
                  max_tokens=_summary_max_tokens(e["target_words"]))
        for key, e in by_key.items()
    ]
    out = client.generate_batch(
        items, model=model_tag, options={"temperature": temperature},
    )
    cache_dir = Path(cache_dir)
    for key, e in by_key.items():
        if key not in out:
            raise RuntimeError(
                f"batch summary missing for custom_id={key} "
                f"(docs {e['indices']}); some requests errored or expired."
            )
        text = out[key].strip()
        _write_summary_cache(
            cache_dir / f"{key}.txt", cache_dir / f"{key}.json", text,
            prompt_version=prompt_version, model_tag=model_tag,
            doc_sha1=_doc_sha1(e["doc"]), target_words=e["target_words"],
            source_words=e["source_words"], fraction=fraction,
            temperature=temperature, seed=seed,
        )
        for i in e["indices"]:
            results[i] = SummaryResult(
                text, key, False, e["target_words"], e["source_words"]
            )
