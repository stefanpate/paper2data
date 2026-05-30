"""Build sklearn Pipelines from hydra configs."""

from __future__ import annotations

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.pipeline import Pipeline


def is_precomputed_featurizer(featurizer_cfg: DictConfig) -> bool:
    return bool(featurizer_cfg.get("precomputed", False))


def _coerce_tuple_params(estimator):
    """Convert list-valued constructor params to tuples in place.

    Sequence hyperparameters (e.g. TfidfVectorizer.ngram_range) come back from
    hydra/OmegaConf as lists, but sklearn requires tuples. The grid search masks
    this because the param grid always overrides such keys with coerced tuples;
    when fitting default hyperparameters directly the raw lists would otherwise
    fail sklearn's parameter validation.
    """
    updates = {
        key: tuple(val)
        for key, val in estimator.get_params(deep=False).items()
        if isinstance(val, (list, ListConfig))
    }
    if updates:
        estimator.set_params(**updates)
    return estimator


def build_pipeline(featurizer_cfg: DictConfig, model_cfg: DictConfig) -> Pipeline:
    """Compose a classifier-only sklearn Pipeline.

    Featurization (TF-IDF or sentence embeddings) is performed once on the full
    corpus in ``train.run`` and fed to this pipeline as pre-computed feature
    vectors, so the pipeline has a single ``clf`` step regardless of featurizer.
    ``featurizer_cfg`` is accepted for signature symmetry but unused.
    """
    clf = _coerce_tuple_params(instantiate(model_cfg.estimator))
    return Pipeline([("clf", clf)])


def build_param_grid(
    featurizer_cfg: DictConfig, model_cfg: DictConfig
) -> dict[str, list]:
    """Build the GridSearchCV grid from the model config only.

    Featurizers are fit once on the full corpus with fixed hyperparameters
    (outside the CV loop), so only classifier hyperparameters are tuned.
    Tuple-valued model hyperparameters come back from OmegaConf as lists;
    sklearn requires tuples, so coerce list-of-lists entries here.
    """
    grid: dict[str, list] = {}
    raw = OmegaConf.to_container(model_cfg.get("param_grid", {}), resolve=True) or {}
    for key, values in raw.items():
        grid[key] = [tuple(v) if isinstance(v, list) else v for v in values]
    return grid
