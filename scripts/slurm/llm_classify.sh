#!/usr/bin/env bash
#SBATCH --job-name=p2d-llm-classify
#SBATCH --account=p30041            # TODO: set Quest allocation
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/llm_classify-%j.out
#SBATCH --error=logs/llm_classify-%j.err
#
# LLM zero-shot classification via a node-local ollama server.
#
# Submit:
#   sbatch scripts/slurm/llm_classify.sh llm=qwen25_7b fraction=0.25
#   sbatch scripts/slurm/llm_classify.sh -m llm=qwen25_7b fraction=0.1,0.25,0.5,0.75,1.0

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

echo "[llm_classify] node=$SLURMD_NODENAME gpu=$CUDA_VISIBLE_DEVICES"
echo "[llm_classify] OLLAMA_HOST=$OLLAMA_HOST tag=$OLLAMA_MODEL_TAG"
echo "[llm_classify] args: $*"

uv run python scripts/llm_classify.py "$@"
