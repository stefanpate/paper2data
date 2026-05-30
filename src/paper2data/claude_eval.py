"""Incremental, per-sample Claude evaluation on 20NG (batched Messages API).

Unlike `llm_classify.run` (which scores the whole test set and writes only
aggregate metrics), this keeps a growable, **append-only** store of individual
predictions. Bump `test_per_category` between runs to send more samples:

    uv run python scripts/claude_eval.py test_per_category=2   # 2 per class
    uv run python scripts/claude_eval.py test_per_category=5   # sends only +3/class

The test-sample selection is *prefix-stable* (growing the count only ever appends
docs), and the store is deduplicated by test index, so a document is never
summarized or classified twice — you never re-send a test sample. The store
(`predictions.jsonl`) holds the actual predicted label and the summary text for
each sample, which the companion notebook (`notebooks/claude_eval.ipynb`) reads.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from paper2data.data import load_twenty_newsgroups
from paper2data.few_shot import build_examples, render_examples_block
from paper2data.llm_classify import (
    _build_classification_model,
    _metrics,
    classify_corpus,
)
from paper2data.llm_providers import make_provider
from paper2data.llm_summaries import _doc_sha1, summarize_corpus

log = logging.getLogger(__name__)


def select_test_indices(y: np.ndarray, n_per_category: int, seed: int) -> list[int]:
    """Prefix-stable selection of `n_per_category` test indices per class.

    For a fixed (y, seed), increasing `n_per_category` only ever APPENDS indices:
    the first k picks for any class are identical whether you ask for k or k+m.
    Each class draws from its own deterministic permutation (seeded by
    (seed, label)), so growing the evaluated set never reshuffles earlier picks —
    which is what lets you add samples without re-sending the old ones.
    """
    picks: list[int] = []
    for label in np.unique(y):
        cand = np.flatnonzero(y == label)
        if len(cand) < n_per_category:
            raise ValueError(
                f"class {int(label)} has only {len(cand)} test docs "
                f"but test_per_category={n_per_category}"
            )
        perm = np.random.default_rng([seed, int(label)]).permutation(cand)
        picks.extend(int(i) for i in perm[:n_per_category])
    return sorted(picks)


def load_store(path: str | Path) -> list[dict]:
    """Read the append-only predictions store (one JSON object per line)."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _append_store(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _store_metrics(store: list[dict], target_names: list[str]) -> dict:
    """Aggregate metrics over the whole store (parseable predictions only)."""
    scored = [r for r in store if r["pred_idx"] is not None]
    y_true = np.array([r["true_idx"] for r in scored], dtype=int)
    y_pred = np.array([r["pred_idx"] for r in scored], dtype=int)
    out = {
        "n_total": len(store),
        "n_scored": len(scored),
        "n_unparseable": len(store) - len(scored),
    }
    if len(scored):
        out.update(_metrics(y_true, y_pred))
    else:
        out.update({"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0})
    return out


def run(cfg: DictConfig) -> dict:
    store_dir = Path(cfg.store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    pred_path = store_dir / "predictions.jsonl"
    OmegaConf.save(cfg, store_dir / "config.yaml")

    ds = load_twenty_newsgroups(
        subset=cfg.data.subset,
        remove=tuple(cfg.data.remove),
        categories=OmegaConf.to_container(cfg.data.categories) if cfg.data.categories else None,
    )
    X = np.asarray(ds.X, dtype=object)
    y = ds.y
    target_names = list(ds.target_names)
    name_to_idx = {c: i for i, c in enumerate(target_names)}

    # Same split as train.py / llm_classify — must stay bit-identical so test
    # indices are comparable across the whole project.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )

    selected = select_test_indices(y_test, int(cfg.test_per_category), int(cfg.seed))
    store = load_store(pred_path)
    done = {int(r["test_idx"]) for r in store}
    new_idx = [i for i in selected if i not in done]

    log.info(
        "provider=%s store=%s: %d selected (%d/class), %d already done, %d new",
        cfg.llm.name, pred_path, len(selected), int(cfg.test_per_category),
        len(done), len(new_idx),
    )

    if new_idx:
        client = make_provider(cfg.llm)
        docs = [X_test[i] for i in new_idx]

        # --- Summarize new docs (batched; reuses the on-disk summary cache) ---
        summaries = summarize_corpus(
            docs,
            fraction=float(cfg.fraction),
            model_tag=cfg.llm.tag,
            cache_dir=cfg.summary_cache_dir,
            num_ctx=int(getattr(cfg.llm, "summary_num_ctx", 16384)),
            temperature=cfg.llm.temperature,
            seed=cfg.llm.seed,
            prompt_template=cfg.prompts.summary,
            prompt_version=cfg.prompts.version,
            client=client,
        )
        texts = [s.text for s in summaries]

        # --- Few-shot examples (optional, from the train split) ---------------
        examples_block = ""
        n_per_cat = int(cfg.few_shot.n_per_category)
        if n_per_cat > 0:
            examples = build_examples(
                X_train, y_train, target_names,
                n_per_category=n_per_cat,
                fraction=float(cfg.fraction),
                model_tag=cfg.llm.tag,
                summary_cache_dir=cfg.summary_cache_dir,
                fewshot_cache_dir=cfg.fewshot_cache_dir,
                num_ctx=int(getattr(cfg.llm, "summary_num_ctx", 16384)),
                temperature=cfg.llm.temperature,
                seed=cfg.llm.seed,
                prompt_template=cfg.prompts.summary,
                prompt_version=cfg.prompts.version,
                client=client,
            )
            examples_block = render_examples_block(examples)

        # --- Classify new docs (batched) --------------------------------------
        classification_cls = _build_classification_model(target_names)
        schema = classification_cls.model_json_schema()
        categories_block = "\n".join(f"- {c}" for c in target_names)
        t0 = time.perf_counter()
        cats = classify_corpus(
            texts,
            client=client,
            model_tag=cfg.llm.tag,
            schema=schema,
            classification_cls=classification_cls,
            categories_block=categories_block,
            n_categories=len(target_names),
            num_ctx=int(getattr(cfg.llm, "classify_num_ctx", 16384)),
            temperature=cfg.llm.temperature,
            seed=cfg.llm.seed,
            prompt_template=cfg.prompts.classify,
            examples_block=examples_block,
        )
        elapsed = time.perf_counter() - t0

        rows = []
        for i, summary, cat in zip(new_idx, texts, cats):
            true_idx = int(y_test[i])
            pred_idx = name_to_idx[cat] if cat is not None else None
            rows.append({
                "test_idx": int(i),
                "doc_sha1": _doc_sha1(str(X_test[i])),
                "true_idx": true_idx,
                "true_label": target_names[true_idx],
                "pred_label": cat,
                "pred_idx": pred_idx,
                "correct": None if cat is None else bool(pred_idx == true_idx),
                "summary": summary,
                "source_words": len(str(X_test[i]).split()),
                "summary_words": len(summary.split()),
                "fraction": float(cfg.fraction),
                "few_shot_n": n_per_cat,
            })
        _append_store(pred_path, rows)
        store = store + rows
        log.info("Appended %d predictions in %.1fs; usage=%s",
                 len(rows), elapsed, client.usage_summary())
    else:
        log.info("No new samples to send; recomputing metrics over existing store.")

    agg = _store_metrics(store, target_names)

    # Persist in the same shape llm_classify.run writes (see
    # paper2data/llm_classify.py) so data_efficiency.ipynb picks Claude runs up
    # alongside the Qwen runs: it needs `kind` and metrics nested under `test`.
    # The flat store counts are retained at top level for notebooks/claude_eval.ipynb.
    n_per_cat = int(cfg.few_shot.n_per_category)
    metrics = {
        "run_name": cfg.run_name,
        "kind": "llm_few_shot" if n_per_cat > 0 else "llm_zero_shot",
        "model": cfg.llm.name,
        "featurizer": f"llm_summary_{cfg.fraction}_n{n_per_cat}",
        "data": cfg.data.name,
        "n_test": agg["n_total"],  # subset actually evaluated, not the full test set
        "n_classes": len(target_names),
        "test": {
            "accuracy": agg["accuracy"],
            "f1_macro": agg["f1_macro"],
            "f1_weighted": agg["f1_weighted"],
        },
        "prompts": {"name": cfg.prompts.name, "version": cfg.prompts.version},
        "few_shot": {"n_per_category": n_per_cat},
        "llm": {
            "provider": getattr(cfg.llm, "provider", "claude_api"),
            "tag": cfg.llm.tag,
            "fraction": float(cfg.fraction),
            "n_unparseable": agg["n_unparseable"],
        },
        # Flat store counts for the companion per-sample notebook.
        "n_total": agg["n_total"],
        "n_scored": agg["n_scored"],
        "n_unparseable": agg["n_unparseable"],
    }
    (store_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (store_dir / "target_names.json").write_text(json.dumps(target_names))
    log.info(
        "Store holds %d predictions (%d scored): acc=%.4f f1_macro=%.4f unparseable=%d",
        agg["n_total"], agg["n_scored"],
        agg["accuracy"], agg["f1_macro"], agg["n_unparseable"],
    )
    return metrics
