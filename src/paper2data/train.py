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
from paper2data.few_shot import select_example_indices
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

    # Subsample to `n_per_category` training docs per class (stratified),
    # mirroring the few-shot LLM runs so data-efficiency curves share an x-axis.
    # `null`/<=0 means use all available training data.
    n_per_category = cfg.get("n_per_category", None)
    if n_per_category is not None:
        n_per_category = int(n_per_category)
        if n_per_category <= 0:
            raise ValueError(
                f"n_per_category must be a positive integer or null, got {n_per_category}"
            )
        n_full = len(X_train)
        idx = select_example_indices(y_train, n_per_category=n_per_category, seed=cfg.seed)
        X_train = X_train[idx]
        y_train = y_train[idx]
        log.info(
            "Subsampled training set to %d/%d docs (n_per_category=%d)",
            len(X_train), n_full, n_per_category,
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

    cv = StratifiedKFold(
        n_splits=cfg.cv.splits, shuffle=cfg.cv.shuffle, random_state=cfg.seed
    )

    # ---- Hyperparameter selection via k-fold CV on train+val ----------------
    # A single level of cross-validation splits train+val into `cfg.cv.splits`
    # folds to score each grid config. The best config is then refit on all of
    # train+val and evaluated once on the held-out test set. (No nested outer
    # loop: the held-out test split already provides the final estimate.)
    log.info("Grid search over %d-fold CV for HPO", cfg.cv.splits)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=cfg.cv.scoring,
        n_jobs=cfg.n_jobs,
        refit=True,
    )
    search.fit(X_train_feat, y_train)
    best_idx = search.best_index_
    cv_summary = {
        "n_splits": int(cfg.cv.splits),
        "scoring": cfg.cv.scoring,
        "best_score_mean": float(search.cv_results_["mean_test_score"][best_idx]),
        "best_score_std": float(search.cv_results_["std_test_score"][best_idx]),
    }
    log.info("CV %s of best config: %.4f ± %.4f",
             cfg.cv.scoring, cv_summary["best_score_mean"], cv_summary["best_score_std"])

    # ---- Evaluate best model on held-out test -------------------------------
    best_model = search.best_estimator_
    y_test_pred = best_model.predict(X_test_feat)
    test_metrics = _metrics(y_test, y_test_pred)

    log.info("Held-out test f1_macro=%.4f acc=%.4f",
             test_metrics["f1_macro"], test_metrics["accuracy"])

    # ---- Persist ------------------------------------------------------------
    results = {
        "run_name": cfg.run_name,
        "kind": "vector_cv",
        "model": cfg.model.name,
        "featurizer": cfg.featurizer.name,
        "data": cfg.data.name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_per_category": n_per_category,
        "n_classes": int(len(ds.target_names)),
        "best_params": search.best_params_,
        "cv": cv_summary,
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
