#!/usr/bin/env bash
#SBATCH --job-name=p2d-fewshot
#SBATCH --account=p30041           # TODO: set Quest allocation
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/precompute_fewshot-%j.out
#SBATCH --error=logs/precompute_fewshot-%j.err
#
# Pre-compute few-shot example (summary, label) pairs from X_train and write
# the sidecar JSONs under artifacts/_fewshot_cache/. Reuses the summary cache
# warmed by precompute_summaries.sh (same prompt_version), so this is fast
# when summaries already exist.
#
# Submit:
#   sbatch scripts/slurm/precompute_fewshot.sh prompts=few_shot_v1 few_shot.n_per_category=1
#   sbatch scripts/slurm/precompute_fewshot.sh prompts=few_shot_v1 few_shot.n_per_category=1 fractions=[0.25,0.5]
#
# Override OLLAMA_MODEL_TAG to match a non-default `llm=` override.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

module purge
module load ollama/0.12.10

export OLLAMA_MODEL_TAG="${OLLAMA_MODEL_TAG:-qwen2.5:7b-instruct}"

# shellcheck disable=SC1091
source scripts/slurm/_ollama_env.sh
trap stop_ollama EXIT

start_ollama

echo "[precompute_fewshot] node=$SLURMD_NODENAME gpu=$CUDA_VISIBLE_DEVICES"
echo "[precompute_fewshot] OLLAMA_HOST=$OLLAMA_HOST tag=$OLLAMA_MODEL_TAG"
echo "[precompute_fewshot] args: $*"

uv run python scripts/precompute_fewshot.py "$@"
