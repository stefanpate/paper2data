"""Hydra entry point: evaluate an LLM zero-shot classifier on 20NG.

Usage:
    uv run python scripts/llm_classify.py llm=qwen25_7b fraction=0.25
    uv run python scripts/llm_classify.py -m llm=qwen25_7b fraction=0.1,0.25,0.5,0.75,1.0
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from paper2data.llm_classify import run

CONF_DIR = str((Path(__file__).resolve().parents[1] / "conf"))


@hydra.main(version_base=None, config_path=CONF_DIR, config_name="llm_classify")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
