"""Nested cross-validation training for text classifiers on 20NG."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from paper2data.data import load_twenty_newsgroups
from paper2data.embeddings import embed_corpus
from paper2data.pipeline import build_param_grid, build_pipeline, is_precomputed_featurizer

log = logging.getLogger(__name__)

# Models that assume non-negative count-like features — incompatible with
# dense embedding featurizers.
_NON_NEGATIVE_MODELS = {"nb", "complement_nb"}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def _check_compat(featurizer_cfg: DictConfig, model_cfg: DictConfig) -> None:
    if is_precomputed_featurizer(featurizer_cfg) and model_cfg.name in _NON_NEGATIVE_MODELS:
        raise ValueError(
            f"Model '{model_cfg.name}' requires non-negative features and is not "
            f"compatible with dense embedding featurizer '{featurizer_cfg.name}'. "
            f"Use logreg, linear_svm, or random_forest with embedding featurizers."
        )


def run(cfg: DictConfig) -> dict:
    _check_compat(cfg.featurizer, cfg.model)

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
    log.info("Loaded %d documents across %d classes", len(X), len(ds.target_names))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )

    # ---- Featurize ----------------------------------------------------------
    embed_info: dict | None = None
    precomputed = is_precomputed_featurizer(cfg.featurizer)
    if precomputed:
        log.info("Pre-encoding documents with %s", cfg.featurizer.model_id)
        train_emb = embed_corpus(
            list(X_train),
            model_id=cfg.featurizer.model_id,
            cache_dir=cfg.featurizer.cache_dir,
            batch_size=cfg.featurizer.batch_size,
            normalize_embeddings=cfg.featurizer.normalize_embeddings,
            passage_prefix=cfg.featurizer.passage_prefix,
            trust_remote_code=cfg.featurizer.trust_remote_code,
            device=cfg.featurizer.device,
        )
        test_emb = embed_corpus(
            list(X_test),
            model_id=cfg.featurizer.model_id,
            cache_dir=cfg.featurizer.cache_dir,
            batch_size=cfg.featurizer.batch_size,
            normalize_embeddings=cfg.featurizer.normalize_embeddings,
            passage_prefix=cfg.featurizer.passage_prefix,
            trust_remote_code=cfg.featurizer.trust_remote_code,
            device=cfg.featurizer.device,
        )
        X_train_feat = train_emb.vectors
        X_test_feat = test_emb.vectors
        embed_info = {
            "model_id": cfg.featurizer.model_id,
            "dim": int(train_emb.vectors.shape[1]),
            "train_cache_key": train_emb.cache_key,
            "test_cache_key": test_emb.cache_key,
            "train_cache_hit": train_emb.cache_hit,
            "test_cache_hit": test_emb.cache_hit,
            "encode_seconds": train_emb.elapsed_s + test_emb.elapsed_s,
        }
    else:
        X_train_feat = X_train
        X_test_feat = X_test

    pipeline = build_pipeline(cfg.featurizer, cfg.model)
    param_grid = build_param_grid(cfg.featurizer, cfg.model)
    log.info("Param grid: %s", param_grid)

    inner_cv = StratifiedKFold(
        n_splits=cfg.cv.inner_splits, shuffle=cfg.cv.shuffle, random_state=cfg.seed
    )
    outer_cv = StratifiedKFold(
        n_splits=cfg.cv.outer_splits, shuffle=cfg.cv.shuffle, random_state=cfg.seed
    )

    # ---- Nested CV on training data -----------------------------------------
    # Outer loop: unbiased generalization estimate.
    # Inner loop: HPO via GridSearchCV.
    log.info("Running nested CV: %d outer x %d inner folds",
             cfg.cv.outer_splits, cfg.cv.inner_splits)

    fold_records: list[dict] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(outer_cv.split(X_train_feat, y_train)):
        log.info("Outer fold %d/%d", fold_idx + 1, cfg.cv.outer_splits)
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring=cfg.cv.scoring,
            n_jobs=cfg.n_jobs,
            refit=cfg.cv.refit_inner,
        )
        search.fit(X_train_feat[tr_idx], y_train[tr_idx])
        y_va_pred = search.predict(X_train_feat[va_idx])
        fold_metrics = _metrics(y_train[va_idx], y_va_pred)
        fold_records.append({
            "fold": fold_idx,
            "best_params": search.best_params_,
            "best_inner_score": float(search.best_score_),
            **fold_metrics,
        })
        log.info("  best_params=%s  outer %s=%.4f",
                 search.best_params_, cfg.cv.scoring,
                 fold_metrics["f1_macro"])

    nested_summary = {
        metric: {
            "mean": float(np.mean([r[metric] for r in fold_records])),
            "std": float(np.std([r[metric] for r in fold_records])),
        }
        for metric in ("accuracy", "f1_macro", "f1_weighted")
    }
    log.info("Nested CV f1_macro: %.4f ± %.4f",
             nested_summary["f1_macro"]["mean"],
             nested_summary["f1_macro"]["std"])

    # ---- Final HPO on full train, evaluate on held-out test -----------------
    log.info("Refitting on full train with inner CV for final HPO")
    final_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring=cfg.cv.scoring,
        n_jobs=cfg.n_jobs,
        refit=True,
    )
    final_search.fit(X_train_feat, y_train)
    best_model = final_search.best_estimator_
    y_test_pred = best_model.predict(X_test_feat)
    test_metrics = _metrics(y_test, y_test_pred)

    log.info("Held-out test f1_macro=%.4f acc=%.4f",
             test_metrics["f1_macro"], test_metrics["accuracy"])

    # ---- Persist ------------------------------------------------------------
    results = {
        "run_name": cfg.run_name,
        "model": cfg.model.name,
        "featurizer": cfg.featurizer.name,
        "data": cfg.data.name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_classes": int(len(ds.target_names)),
        "best_params": final_search.best_params_,
        "best_inner_score": float(final_search.best_score_),
        "nested_cv": {"folds": fold_records, "summary": nested_summary},
        "test": test_metrics,
        "embedding": embed_info,
    }
    (artifacts_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=str))

    report = classification_report(
        y_test, y_test_pred, target_names=ds.target_names, output_dict=True, zero_division=0
    )
    (artifacts_dir / "classification_report.json").write_text(json.dumps(report, indent=2))

    cm = confusion_matrix(y_test, y_test_pred)
    np.save(artifacts_dir / "confusion_matrix.npy", cm)
    (artifacts_dir / "target_names.json").write_text(json.dumps(ds.target_names))

    joblib.dump(best_model, artifacts_dir / "best_model.joblib")
    log.info("Wrote artifacts to %s", artifacts_dir)
    return results
