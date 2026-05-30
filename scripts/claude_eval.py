"""Hydra entry point: incremental, per-sample Claude eval on 20NG (batched API).

Bump test_per_category between runs to send more samples; previously evaluated
docs are never re-sent. Inspect results with notebooks/claude_eval.ipynb.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    uv run python scripts/claude_eval.py test_per_category=2 fraction=0.25
    uv run python scripts/claude_eval.py test_per_category=5        # sends only +3/class
    uv run python scripts/claude_eval.py llm.use_batch=false        # synchronous (no 24h wait)
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from paper2data.claude_eval import run

CONF_DIR = str((Path(__file__).resolve().parents[1] / "conf"))


@hydra.main(version_base=None, config_path=CONF_DIR, config_name="claude_eval")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
