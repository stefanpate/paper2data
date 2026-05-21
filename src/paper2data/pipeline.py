"""Build sklearn Pipelines from hydra configs."""

from __future__ import annotations

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.pipeline import Pipeline


def is_precomputed_featurizer(featurizer_cfg: DictConfig) -> bool:
    return bool(featurizer_cfg.get("precomputed", False))


def build_pipeline(featurizer_cfg: DictConfig, model_cfg: DictConfig) -> Pipeline:
    """Compose a sklearn Pipeline from hydra config groups.

    If the featurizer is marked ``precomputed: true``, the returned pipeline
    has a single ``clf`` step — the caller is responsible for feeding it
    pre-encoded feature vectors. Otherwise, the featurizer is instantiated as
    the first step (``tfidf``).
    """
    clf = instantiate(model_cfg.estimator)
    if is_precomputed_featurizer(featurizer_cfg):
        return Pipeline([("clf", clf)])
    tfidf = instantiate(featurizer_cfg.estimator)
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def build_param_grid(
    featurizer_cfg: DictConfig, model_cfg: DictConfig
) -> dict[str, list]:
    """Merge featurizer + model param grids into a single dict for GridSearchCV.

    Tuple-valued hyperparameters (e.g. ngram_range) come back from OmegaConf as
    lists; sklearn requires tuples, so coerce list-of-lists entries here.
    """
    grid: dict[str, list] = {}
    cfgs = [model_cfg]
    if not is_precomputed_featurizer(featurizer_cfg):
        cfgs.insert(0, featurizer_cfg)
    for cfg in cfgs:
        raw = OmegaConf.to_container(cfg.get("param_grid", {}), resolve=True) or {}
        for key, values in raw.items():
            grid[key] = [tuple(v) if isinstance(v, list) else v for v in values]
    return grid
