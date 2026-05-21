#!/usr/bin/env bash
#SBATCH --job-name=p2d-summaries
#SBATCH --account=p30041           # TODO: set Quest allocation
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/precompute_summaries-%j.out
#SBATCH --error=logs/precompute_summaries-%j.err
#
# Pre-compute LLM summaries via a node-local ollama server.
#
# Submit:
#   sbatch scripts/slurm/precompute_summaries.sh
#   sbatch scripts/slurm/precompute_summaries.sh llm=qwen25_7b fractions=[0.1,0.25]
#   sbatch scripts/slurm/precompute_summaries.sh +smoke=20
#
# Override the model tag pulled into ollama with OLLAMA_MODEL_TAG, e.g.:
#   OLLAMA_MODEL_TAG=qwen2.5:3b-instruct sbatch scripts/slurm/precompute_summaries.sh llm=qwen25_3b

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

module purge
module load ollama/0.12.10                # latest stable on Quest as of writing

# Tag to pre-pull. Defaults to the qwen2.5:7b-instruct used by conf/llm/qwen25_7b.yaml.
# If you pass a different `llm=...` hydra override, set OLLAMA_MODEL_TAG to match.
export OLLAMA_MODEL_TAG="${OLLAMA_MODEL_TAG:-qwen2.5:7b-instruct}"

# shellcheck disable=SC1091
source scripts/slurm/_ollama_env.sh
trap stop_ollama EXIT

start_ollama

echo "[precompute] node=$SLURMD_NODENAME gpu=$CUDA_VISIBLE_DEVICES"
echo "[precompute] OLLAMA_HOST=$OLLAMA_HOST tag=$OLLAMA_MODEL_TAG"
echo "[precompute] args: $*"

uv run python scripts/precompute_summaries.py "$@"
