"""Pre-compute few-shot example pairs for the 20NG train set at multiple fractions.

Warms `artifacts/_summary_cache/` with summaries of the selected training docs
and writes one sidecar JSON per fraction to `artifacts/_fewshot_cache/`.

Usage:
    uv run python scripts/precompute_fewshot.py
    uv run python scripts/precompute_fewshot.py few_shot.n_per_category=2
    uv run python scripts/precompute_fewshot.py fractions=[0.25,0.5] llm=qwen25_7b
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from paper2data.data import load_twenty_newsgroups
from paper2data.few_shot import build_examples

log = logging.getLogger(__name__)

CONF_DIR = str((Path(__file__).resolve().parents[1] / "conf"))


@hydra.main(version_base=None, config_path=CONF_DIR, config_name="llm_classify")
def main(cfg: DictConfig) -> None:
    n_per_cat = int(cfg.few_shot.n_per_category)
    if n_per_cat <= 0:
        log.warning("few_shot.n_per_category=%d — nothing to precompute. "
                    "Override at the CLI, e.g. few_shot.n_per_category=1.",
                    n_per_cat)
        return

    ds = load_twenty_newsgroups(
        subset=cfg.data.subset,
        remove=tuple(cfg.data.remove),
        categories=OmegaConf.to_container(cfg.data.categories) if cfg.data.categories else None,
    )
    X = np.asarray(ds.X, dtype=object)
    y = ds.y
    target_names = list(ds.target_names)

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )
    log.info("Train pool: %d docs, %d classes; n_per_category=%d",
             len(X_train), len(target_names), n_per_cat)

    for frac in cfg.fractions:
        frac = float(frac)
        log.info("=== fraction=%.2f ===", frac)
        build_examples(
            X_train, y_train, target_names,
            n_per_category=n_per_cat,
            fraction=frac,
            model_tag=cfg.llm.tag,
            summary_cache_dir=cfg.summary_cache_dir,
            fewshot_cache_dir=cfg.fewshot_cache_dir,
            num_ctx=cfg.llm.summary_num_ctx,
            temperature=cfg.llm.temperature,
            seed=cfg.llm.seed,
            prompt_template=cfg.prompts.summary,
            prompt_version=cfg.prompts.version,
        )

    log.info("Done. Sidecars in: %s", cfg.fewshot_cache_dir)


if __name__ == "__main__":
    main()
