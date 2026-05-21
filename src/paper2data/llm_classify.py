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
from paper2data.llm_summaries import summarize_corpus

log = logging.getLogger(__name__)

CLASSIFY_PROMPT = """You are classifying a Usenet newsgroup post into exactly one of \
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
) -> str | None:
    prompt = CLASSIFY_PROMPT.format(
        n_categories=n_categories,
        categories_block=categories_block,
        text=text,
    )
    resp = client.chat(
        model=model_tag,
        messages=[{"role": "user", "content": prompt}],
        format=schema,
        options={"temperature": temperature, "num_ctx": num_ctx, "seed": seed},
    )
    content = resp["message"]["content"] if isinstance(resp, dict) else resp.message.content
    try:
        return classification_cls.model_validate_json(content).category
    except ValidationError:
        log.warning("Unparseable classification response: %r", content[:200])
        return None


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
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )
    log.info("Held-out test set: %d docs, %d classes", len(X_test), len(target_names))

    if cfg.smoke:
        n = int(cfg.smoke)
        log.info("SMOKE MODE: truncating to first %d docs", n)
        X_test = X_test[:n]
        y_test = y_test[:n]

    # --- Summarize ---------------------------------------------------------
    log.info("Summarizing test set at fraction=%.2f", cfg.fraction)
    summaries = summarize_corpus(
        list(X_test),
        fraction=float(cfg.fraction),
        model_tag=cfg.llm.tag,
        cache_dir=cfg.summary_cache_dir,
        num_ctx=cfg.llm.summary_num_ctx,
        temperature=cfg.llm.temperature,
        seed=cfg.llm.seed,
    )
    texts = [s.text for s in summaries]
    mean_summary_words = float(np.mean([len(t.split()) for t in texts]))

    # --- Classify ----------------------------------------------------------
    import ollama
    client = ollama.Client()
    classification_cls = _build_classification_model(target_names)
    schema = classification_cls.model_json_schema()
    categories_block = "\n".join(f"- {c}" for c in target_names)
    name_to_idx = {c: i for i, c in enumerate(target_names)}

    log.info("Classifying %d docs with %s", len(texts), cfg.llm.tag)
    preds: list[int] = []
    n_unparseable = 0
    t0 = time.perf_counter()

    try:
        from tqdm import tqdm
        iterator = tqdm(texts, desc="classify")
    except ImportError:
        iterator = texts

    for text in iterator:
        cat = classify_one(
            text,
            client=client,
            model_tag=cfg.llm.tag,
            schema=schema,
            classification_cls=classification_cls,
            categories_block=categories_block,
            n_categories=len(target_names),
            num_ctx=cfg.llm.classify_num_ctx,
            temperature=cfg.llm.temperature,
            seed=cfg.llm.seed,
        )
        if cat is None:
            n_unparseable += 1
            preds.append(-1)
        else:
            preds.append(name_to_idx[cat])
    elapsed = time.perf_counter() - t0

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
        "kind": "llm_zero_shot",
        "model": cfg.llm.name,
        "featurizer": f"llm_summary_{cfg.fraction}",
        "data": cfg.data.name,
        "n_test": int(len(X_test)),
        "n_classes": int(len(target_names)),
        "test": test_metrics,
        "llm": {
            "tag": cfg.llm.tag,
            "fraction": float(cfg.fraction),
            "n_unparseable": int(n_unparseable),
            "mean_summary_words": mean_summary_words,
            "elapsed_s": elapsed,
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
