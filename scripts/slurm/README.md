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
| `precompute_summaries.sh` | GPU + ollama | Pre-generate LLM summaries of the **test** docs (`scripts/precompute_summaries.py`). |
| `precompute_fewshot.sh` | GPU + ollama | Pre-generate few-shot example (summary, label) pairs from **train** docs (`scripts/precompute_fewshot.py`). |
| `llm_classify.sh` | GPU + ollama | LLM classification — zero-shot or few-shot (`scripts/llm_classify.py`). |

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

## Prompts and few-shot

Prompt templates live under `conf/prompts/`. The two variants currently
checked in:

- `prompts=zero_shot_v1` (default) — no in-context examples.
- `prompts=few_shot_v1` — has an `{examples_block}` placeholder; pair it
  with `few_shot.n_per_category=N` (N ≥ 1).

Examples are sampled per-class from `X_train` (the **same**
`train_test_split` `train.py` uses, so the LLM-vs-vector comparison is
apples-to-apples), summarized at the requested `fraction`, persisted to
`artifacts/_fewshot_cache/<key>.json`, and rendered into the classify
prompt as `POST … RESPONSE: {"category": "<label>"}` pairs that mirror
the schema-constrained output.

`run_name` now encodes the prompt variant and few-shot count, so zero-
and few-shot artifacts don't collide:

```
${llm.name}_${prompts.name}_n${few_shot.n_per_category}_summary${fraction}_${data.name}
# e.g. qwen25_7b_few_shot_v1_n1_summary0.25_20ng
```

`metrics.json` additionally records `prompts.{name,version}` and
`few_shot.{n_per_category, example_train_indices}` for reproducibility.

### Typical few-shot workflow

```bash
# 1. (optional but recommended) warm the summary cache for the test set.
sbatch scripts/slurm/precompute_summaries.sh

# 2. Warm the few-shot sidecars across fractions.
sbatch scripts/slurm/precompute_fewshot.sh \
    prompts=few_shot_v1 few_shot.n_per_category=1

# 3. Run classification — multirun across fractions.
sbatch scripts/slurm/llm_classify.sh -m \
    prompts=few_shot_v1 few_shot.n_per_category=1 \
    fraction=0.1,0.25,0.5,0.75,1.0
```

### Zero-shot sweep (unchanged)

```bash
sbatch scripts/slurm/precompute_summaries.sh
sbatch scripts/slurm/llm_classify.sh -m fraction=0.1,0.25,0.5,0.75,1.0
```

### Sweeping `n_per_category`

```bash
sbatch scripts/slurm/precompute_fewshot.sh \
    prompts=few_shot_v1 few_shot.n_per_category=2
sbatch scripts/slurm/llm_classify.sh -m \
    prompts=few_shot_v1 few_shot.n_per_category=1,2,4 fraction=0.25
```
Run `precompute_fewshot.sh` once per `n_per_category` you intend to
classify with — the sidecar is keyed on it.

## Caches and reuse

- `artifacts/_summary_cache/` — keyed on
  `(model_tag, doc_sha1, target_words, temperature, seed, prompt_version)`.
  Training- and test-doc summaries share this cache. Bump
  `prompts.version` only when you edit `summary:` in the prompt yaml.
- `artifacts/_fewshot_cache/` — keyed on
  `(model_tag, prompt_version, fraction, n_per_category, seed)`. Cheap to
  regenerate from the summary cache; safe to `rm -rf` if you change the
  selection logic.

## Smoke testing

Most scripts honor `+smoke=N` to truncate inputs and produce a fast
sanity run, e.g.:

```bash
sbatch scripts/slurm/llm_classify.sh \
    prompts=few_shot_v1 few_shot.n_per_category=1 fraction=0.25 +smoke=20
```
