# Quest Slurm scripts

Submit from the repo root so `$SLURM_SUBMIT_DIR` resolves correctly.

## One-time setup on Quest

1. Clone the repo into your home or project directory.
2. Install `uv` (project package manager) if not already on PATH:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   ```
3. Sync the venv once on a login node (or in an interactive job):
   ```bash
   uv sync
   ```
4. Edit each `*.sh` and replace `<ACCOUNT>` with your Quest allocation
   (e.g. `p12345`, `b1234`).

The ollama module is loaded inside the GPU jobs — no manual `module load`
needed beforehand. Pulled model blobs land in `$SCRATCH/ollama_models`
(falls back to `$HOME/ollama_models`); set `OLLAMA_MODELS` to override.

## Jobs

| Script | Backend | Purpose |
| --- | --- | --- |
| `train.sh` | CPU | Nested CV over sklearn pipelines (`scripts/train.py`). |
| `precompute_summaries.sh` | GPU + ollama | Pre-generate LLM summaries (`scripts/precompute_summaries.py`). |
| `llm_classify.sh` | GPU + ollama | Zero-shot LLM classification (`scripts/llm_classify.py`). |

All extra args are forwarded to the python entrypoint, so hydra
overrides and `-m` multirun work:

```bash
sbatch scripts/slurm/train.sh -m model=nb,logreg,linear_svm
sbatch scripts/slurm/llm_classify.sh -m llm=qwen25_7b fraction=0.1,0.25,0.5
```

If you pass a non-default `llm=` override, set `OLLAMA_MODEL_TAG` to the
matching ollama tag so the right weights get pulled — e.g.:

```bash
OLLAMA_MODEL_TAG=qwen2.5:3b-instruct \
  sbatch scripts/slurm/llm_classify.sh llm=qwen25_3b
```
