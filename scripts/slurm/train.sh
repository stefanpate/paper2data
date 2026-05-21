#!/usr/bin/env bash
#SBATCH --job-name=p2d-train
#SBATCH --account=p30041            # TODO: set Quest allocation
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err
#
# Nested-CV training. Pure CPU/sklearn — no ollama needed.
#
# Submit:
#   sbatch scripts/slurm/train.sh model=nb
#   sbatch scripts/slurm/train.sh -m model=nb,logreg,linear_svm
# Anything after the script name is forwarded to the python entrypoint
# (so hydra overrides and -m multirun all work).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# uv is the project's package manager — it lives in the user env, not a module.
# If uv isn't on PATH on your login/compute node, add it here, e.g.:
#   export PATH="$HOME/.local/bin:$PATH"

echo "[train] node=$SLURMD_NODENAME cpus=$SLURM_CPUS_PER_TASK"
echo "[train] args: $*"

uv run python scripts/train.py "$@"
