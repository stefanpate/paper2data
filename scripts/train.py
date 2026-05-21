"""Hydra entry point for nested-CV training.

Usage:
    uv run python scripts/train.py model=nb
    uv run python scripts/train.py -m model=nb,logreg,linear_svm
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from paper2data.train import run

CONF_DIR = str((Path(__file__).resolve().parents[1] / "conf"))


@hydra.main(version_base=None, config_path=CONF_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
