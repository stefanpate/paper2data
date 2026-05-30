#!/usr/bin/env bash
#SBATCH --job-name=p2d-llm-claude
#SBATCH --account=p30041            # TODO: set Quest allocation
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/llm_classify_claude-%j.out
#SBATCH --error=logs/llm_classify_claude-%j.err
#
# LLM classification via Claude (Anthropic Messages API + Batch) — no GPU, no
# ollama server. Pay-per-token; requires ANTHROPIC_API_KEY in the environment.
# Batch requests are async (<=24h), so keep --time generous.
#
# Submit (export your key first; don't hard-code it here):
#   ANTHROPIC_API_KEY=sk-ant-... sbatch scripts/slurm/llm_classify_claude.sh \
#       test_per_category=2 fraction=0.25
#   ANTHROPIC_API_KEY=sk-ant-... sbatch scripts/slurm/llm_classify_claude.sh -m \
#       fraction=0.1,0.25,0.5,0.75,1.0
# (llm=claude_sonnet is the default in conf/claude_eval.yaml. test_per_category
#  selection is prefix-stable, so re-submitting with a larger value only sends
#  the new samples.)

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "[llm_classify_claude] FATAL: ANTHROPIC_API_KEY not set" >&2
  exit 1
fi

echo "[llm_classify_claude] node=$SLURMD_NODENAME"
echo "[llm_classify_claude] args: $*"

uv run python scripts/claude_eval.py "$@"
