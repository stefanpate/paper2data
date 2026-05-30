"""Zero-shot classification of 20NG documents (or their LLM summaries) via Ollama.

Uses JSON-schema-constrained decoding (Ollama `format=` arg) so the model's
response is guaranteed to be one of the 20 target categories.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Literal

import numpy as np
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ValidationError, create_model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from paper2data.data import load_twenty_newsgroups
from paper2data.few_shot import build_examples, render_examples_block
from paper2data.llm_summaries import summarize_corpus

log = logging.getLogger(__name__)

# Fallback default — the canonical prompt lives in `conf/prompts/zero_shot_v1.yaml`
# and is threaded through `run()` via cfg.prompts.classify.
DEFAULT_CLASSIFY_PROMPT = """You are classifying a Usenet newsgroup post into exactly one of \
{n_categories} categories.

Categories:
{categories_block}

POST:
{text}

Choose the single best category. Respond with JSON matching the schema."""


def _build_classification_model(target_names: list[str]) -> type[BaseModel]:
    """Build a pydantic model whose `category` field is a Literal[*target_names]."""
    return create_model(
        "Classification",
        category=(Literal[tuple(target_names)], ...),  # type: ignore[valid-type]
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def _next_pow2(n: int) -> int:
    return 1 << max(0, n - 1).bit_length()


def _build_classify_prompt(
    prompt_template: str, *, n_categories: int, categories_block: str,
    text: str, examples_block: str = "",
) -> str:
    fields = {
        "n_categories": n_categories,
        "categories_block": categories_block,
        "text": text,
    }
    if "{examples_block}" in prompt_template:
        fields["examples_block"] = examples_block
    return prompt_template.format(**fields)


def classify_one(
    text: str,
    *,
    client,
    model_tag: str,
    schema: dict,
    classification_cls: type[BaseModel],
    categories_block: str,
    n_categories: int,
    num_ctx: int,
    temperature: float,
    seed: int,
    prompt_template: str = DEFAULT_CLASSIFY_PROMPT,
    examples_block: str = "",
    tokenizer=None,
    max_num_ctx: int | None = None,
    response_headroom: int = 64,
) -> str | None:
    prompt = _build_classify_prompt(
        prompt_template, n_categories=n_categories,
        categories_block=categories_block, text=text, examples_block=examples_block,
    )

    if tokenizer is not None:
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        required = prompt_tokens + response_headroom
        if required > num_ctx:
            ceiling = max_num_ctx if max_num_ctx is not None else required
            grown = min(_next_pow2(required), ceiling)
            if grown < required:
                raise RuntimeError(
                    f"classify prompt needs {required} tokens "
                    f"({prompt_tokens} prompt + {response_headroom} headroom) "
                    f"but max_classify_num_ctx={ceiling}. "
                    f"Lower few_shot.n_per_category, lower fraction, or raise the ceiling."
                )
            log.warning(
                "classify: prompt is %d tokens; growing num_ctx %d -> %d",
                prompt_tokens, num_ctx, grown,
            )
            num_ctx = grown

    resp = client.chat(
        model=model_tag,
        messages=[{"role": "user", "content": prompt}],
        format=schema,
        options={"temperature": temperature, "num_ctx": num_ctx, "seed": seed},
        max_tokens=_CLASSIFY_MAX_TOKENS,
    )
    content = resp["message"]["content"] if isinstance(resp, dict) else resp.message.content
    try:
        return classification_cls.model_validate_json(content).category
    except ValidationError:
        log.warning("Unparseable classification response: %r", content[:200])
        return None


_CLASSIFY_MAX_TOKENS = 64


def classify_corpus(
    texts: list[str],
    *,
    client,
    model_tag: str,
    schema: dict,
    classification_cls: type[BaseModel],
    categories_block: str,
    n_categories: int,
    num_ctx: int,
    temperature: float,
    seed: int,
    prompt_template: str = DEFAULT_CLASSIFY_PROMPT,
    examples_block: str = "",
    tokenizer=None,
    max_num_ctx: int | None = None,
    show_progress: bool = True,
) -> list[str | None]:
    """Classify many docs, routing to the provider's Batch API when available.

    Returns one category name (or None for unparseable) per input, in order.
    Non-batch providers loop `classify_one`, serially or concurrently per
    `client.concurrency`.
    """
    n = len(texts)
    preds: list[str | None] = [None] * n

    if client.supports_batch():
        from paper2data.llm_providers import BatchItem
        items = []
        for i, text in enumerate(texts):
            prompt = _build_classify_prompt(
                prompt_template, n_categories=n_categories,
                categories_block=categories_block, text=text,
                examples_block=examples_block,
            )
            items.append(BatchItem(
                custom_id=f"doc{i}",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=_CLASSIFY_MAX_TOKENS,
            ))
        out = client.chat_batch(
            items, model=model_tag, format=schema,
            options={"temperature": temperature},
        )
        for i in range(n):
            content = out.get(f"doc{i}")
            if not content:
                continue
            try:
                preds[i] = classification_cls.model_validate_json(content).category
            except ValidationError:
                log.warning("Unparseable classification response: %r", content[:200])
        return preds

    def _one(i: int) -> tuple[int, str | None]:
        return i, classify_one(
            texts[i], client=client, model_tag=model_tag, schema=schema,
            classification_cls=classification_cls, categories_block=categories_block,
            n_categories=n_categories, num_ctx=num_ctx, temperature=temperature,
            seed=seed, prompt_template=prompt_template, examples_block=examples_block,
            tokenizer=tokenizer, max_num_ctx=max_num_ctx,
        )

    concurrency = int(getattr(client, "concurrency", 1) or 1)
    indices = range(n)
    if show_progress:
        try:
            from tqdm import tqdm
            indices = tqdm(list(indices), desc="classify")
        except ImportError:
            pass

    if concurrency <= 1:
        for i in indices:
            _, preds[i] = _one(i)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for idx, cat in ex.map(_one, indices):
                preds[idx] = cat
    return preds


def run(cfg: DictConfig) -> dict:
    artifacts_dir = Path(cfg.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, artifacts_dir / "config.yaml")

    log.info("Loading 20NG (subset=%s)", cfg.data.subset)
    ds = load_twenty_newsgroups(
        subset=cfg.data.subset,
        remove=tuple(cfg.data.remove),
        categories=OmegaConf.to_container(cfg.data.categories) if cfg.data.categories else None,
    )
    X = np.asarray(ds.X, dtype=object)
    y = ds.y
    target_names = ds.target_names

    # Same split as src/paper2data/train.py — must stay bit-identical.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )
    log.info("Held-out test set: %d docs, %d classes", len(X_test), len(target_names))

    if cfg.smoke:
        n = int(cfg.smoke)
        log.info("SMOKE MODE: truncating to first %d docs", n)
        X_test = X_test[:n]
        y_test = y_test[:n]

    # --- Provider ----------------------------------------------------------
    from paper2data.llm_providers import make_provider
    provider = str(getattr(cfg.llm, "provider", "ollama"))
    log.info("LLM provider=%s tag=%s", provider, cfg.llm.tag)
    client = make_provider(cfg.llm)
    summary_num_ctx = int(getattr(cfg.llm, "summary_num_ctx", 16384))

    # --- Summarize ---------------------------------------------------------
    log.info("Summarizing test set at fraction=%.2f", cfg.fraction)
    summaries = summarize_corpus(
        list(X_test),
        fraction=float(cfg.fraction),
        model_tag=cfg.llm.tag,
        cache_dir=cfg.summary_cache_dir,
        num_ctx=summary_num_ctx,
        temperature=cfg.llm.temperature,
        seed=cfg.llm.seed,
        prompt_template=cfg.prompts.summary,
        prompt_version=cfg.prompts.version,
        client=client,
    )
    texts = [s.text for s in summaries]
    mean_summary_words = float(np.mean([len(t.split()) for t in texts]))

    # --- Build few-shot examples (if requested) ----------------------------
    n_per_cat = int(cfg.few_shot.n_per_category)
    examples = []
    examples_block = ""
    if n_per_cat > 0:
        examples = build_examples(
            X_train, y_train, list(target_names),
            n_per_category=n_per_cat,
            fraction=float(cfg.fraction),
            model_tag=cfg.llm.tag,
            summary_cache_dir=cfg.summary_cache_dir,
            fewshot_cache_dir=cfg.fewshot_cache_dir,
            num_ctx=summary_num_ctx,
            temperature=cfg.llm.temperature,
            seed=cfg.llm.seed,
            prompt_template=cfg.prompts.summary,
            prompt_version=cfg.prompts.version,
            client=client,
        )
        examples_block = render_examples_block(examples)
        if "{examples_block}" not in cfg.prompts.classify:
            log.warning(
                "few_shot.n_per_category=%d but prompts.%s has no "
                "{examples_block} placeholder — examples will be discarded.",
                n_per_cat, cfg.prompts.name,
            )

    # --- Classify ----------------------------------------------------------
    classification_cls = _build_classification_model(target_names)
    schema = classification_cls.model_json_schema()
    categories_block = "\n".join(f"- {c}" for c in target_names)
    name_to_idx = {c: i for i, c in enumerate(target_names)}

    # The num_ctx auto-grow + tokenizer length check is Ollama-specific (Claude
    # manages its own 200K/1M context). Only enable it for the Ollama provider.
    tokenizer = None
    classify_num_ctx = int(getattr(cfg.llm, "classify_num_ctx", 16384))
    max_classify_num_ctx = classify_num_ctx
    if provider == "ollama":
        hf_tok = getattr(cfg.llm, "hf_tokenizer", None)
        if hf_tok:
            from transformers import AutoTokenizer
            log.info("classify: loading tokenizer %s for prompt-length checks", hf_tok)
            tokenizer = AutoTokenizer.from_pretrained(hf_tok)
        max_classify_num_ctx = int(getattr(cfg.llm, "max_classify_num_ctx", classify_num_ctx))

    log.info("Classifying %d docs with %s", len(texts), cfg.llm.tag)
    t0 = time.perf_counter()
    cats = classify_corpus(
        texts,
        client=client,
        model_tag=cfg.llm.tag,
        schema=schema,
        classification_cls=classification_cls,
        categories_block=categories_block,
        n_categories=len(target_names),
        num_ctx=classify_num_ctx,
        temperature=cfg.llm.temperature,
        seed=cfg.llm.seed,
        prompt_template=cfg.prompts.classify,
        examples_block=examples_block,
        tokenizer=tokenizer,
        max_num_ctx=max_classify_num_ctx,
    )
    elapsed = time.perf_counter() - t0

    preds = [name_to_idx[c] if c is not None else -1 for c in cats]
    n_unparseable = sum(1 for c in cats if c is None)
    y_pred = np.asarray(preds)
    valid_mask = y_pred != -1
    if not valid_mask.all():
        log.warning("%d unparseable predictions; excluding from metrics.", (~valid_mask).sum())

    test_metrics = _metrics(y_test[valid_mask], y_pred[valid_mask])
    log.info("Test f1_macro=%.4f acc=%.4f (n=%d, unparseable=%d)",
             test_metrics["f1_macro"], test_metrics["accuracy"],
             int(valid_mask.sum()), n_unparseable)

    # --- Persist (mirror existing artifact layout) -------------------------
    results = {
        "run_name": cfg.run_name,
        "kind": "llm_few_shot" if n_per_cat > 0 else "llm_zero_shot",
        "model": cfg.llm.name,
        "featurizer": f"llm_summary_{cfg.fraction}_n{n_per_cat}",
        "data": cfg.data.name,
        "n_test": int(len(X_test)),
        "n_classes": int(len(target_names)),
        "test": test_metrics,
        "prompts": {"name": cfg.prompts.name, "version": cfg.prompts.version},
        "few_shot": {
            "n_per_category": n_per_cat,
            "example_train_indices": [e.train_idx for e in examples],
        },
        "llm": {
            "provider": provider,
            "tag": cfg.llm.tag,
            "fraction": float(cfg.fraction),
            "n_unparseable": int(n_unparseable),
            "mean_summary_words": mean_summary_words,
            "elapsed_s": elapsed,
            "usage": client.usage_summary(),
        },
    }
    (artifacts_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=str))

    report = classification_report(
        y_test[valid_mask], y_pred[valid_mask],
        labels=list(range(len(target_names))),
        target_names=target_names, output_dict=True, zero_division=0,
    )
    (artifacts_dir / "classification_report.json").write_text(json.dumps(report, indent=2))

    cm = confusion_matrix(y_test[valid_mask], y_pred[valid_mask],
                          labels=list(range(len(target_names))))
    np.save(artifacts_dir / "confusion_matrix.npy", cm)
    (artifacts_dir / "target_names.json").write_text(json.dumps(target_names))

    log.info("Wrote artifacts to %s", artifacts_dir)
    return results
