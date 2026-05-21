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
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Bump this string when the prompt template changes — invalidates cache.
PROMPT_VERSION = "v1"

SUMMARY_PROMPT = """You are summarizing a Usenet newsgroup post for a downstream classifier.
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
               temperature: float, seed: int) -> str:
    payload = json.dumps(
        {
            "v": PROMPT_VERSION,
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
        import ollama
        client = ollama.Client()

    prompt = SUMMARY_PROMPT.format(target_words=target_words, document=doc)
    resp = client.generate(
        model=model_tag,
        prompt=prompt,
        options={
            "temperature": temperature,
            "num_ctx": num_ctx,
            "seed": seed,
        },
    )
    text = resp["response"].strip() if isinstance(resp, dict) else resp.response.strip()

    txt_path.write_text(text)
    meta_path.write_text(json.dumps({
        "prompt_version": PROMPT_VERSION,
        "model_tag": model_tag,
        "doc_sha1": _doc_sha1(doc),
        "target_words": target_words,
        "source_words": source_words,
        "fraction": fraction,
        "temperature": temperature,
        "seed": seed,
    }, indent=2))

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
) -> list[SummaryResult]:
    """Summarize a list of documents. Cache hits are O(disk read); misses call the LLM."""
    import ollama
    client = ollama.Client()

    iterator = enumerate(docs)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc=f"summarize@{fraction}")
        except ImportError:
            pass

    results: list[SummaryResult] = []
    hits = 0
    t0 = time.perf_counter()
    for _, doc in iterator:
        r = summarize_doc(
            doc,
            fraction=fraction,
            model_tag=model_tag,
            cache_dir=cache_dir,
            num_ctx=num_ctx,
            temperature=temperature,
            seed=seed,
            client=client,
        )
        results.append(r)
        if r.cache_hit:
            hits += 1
    elapsed = time.perf_counter() - t0
    log.info(
        "Summarized %d docs at fraction=%.2f (%d cache hits, %d misses) in %.1fs",
        len(docs), fraction, hits, len(docs) - hits, elapsed,
    )
    return results
