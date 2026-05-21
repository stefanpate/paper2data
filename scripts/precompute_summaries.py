"""Pre-compute LLM summaries for the 20NG test set at multiple length fractions.

Usage:
    uv run python scripts/precompute_summaries.py
    uv run python scripts/precompute_summaries.py llm=qwen25_7b fractions=[0.1,0.25]
    uv run python scripts/precompute_summaries.py +smoke=20
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from paper2data.data import load_twenty_newsgroups
from paper2data.llm_summaries import summarize_corpus

log = logging.getLogger(__name__)

CONF_DIR = str((Path(__file__).resolve().parents[1] / "conf"))


@hydra.main(version_base=None, config_path=CONF_DIR, config_name="llm_classify")
def main(cfg: DictConfig) -> None:
    ds = load_twenty_newsgroups(
        subset=cfg.data.subset,
        remove=tuple(cfg.data.remove),
        categories=OmegaConf.to_container(cfg.data.categories) if cfg.data.categories else None,
    )
    X = np.asarray(ds.X, dtype=object)
    y = ds.y

    _, X_test, _, _ = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )
    docs = list(X_test)
    if cfg.smoke:
        docs = docs[: int(cfg.smoke)]
        log.info("SMOKE MODE: %d docs", len(docs))

    log.info("Precomputing summaries for %d docs with %s",
             len(docs), cfg.llm.tag)

    for frac in cfg.fractions:
        frac = float(frac)
        if frac >= 1.0:
            log.info("Skipping fraction=1.0 (raw doc baseline; no LLM call)")
            continue
        log.info("=== fraction=%.2f ===", frac)
        summarize_corpus(
            docs,
            fraction=frac,
            model_tag=cfg.llm.tag,
            cache_dir=cfg.summary_cache_dir,
            num_ctx=cfg.llm.summary_num_ctx,
            temperature=cfg.llm.temperature,
            seed=cfg.llm.seed,
        )

    log.info("Done. Cache dir: %s", cfg.summary_cache_dir)


if __name__ == "__main__":
    main()
