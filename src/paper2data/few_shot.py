"""Few-shot in-context learning support for the LLM classifier.

Picks `n_per_category` documents from the same `X_train` split that `train.py`
uses to fit the vector-based baselines, summarizes them with the same prompt
+ fraction as the test docs (cache-shared via `summarize_corpus`), and renders
them into a prompt block of (POST, RESPONSE-JSON) pairs that mirrors the
schema-constrained output the classifier is forced to emit.

The selected examples are persisted to a sidecar JSON under
`fewshot_cache_dir`, keyed by (model_tag, prompt_version, fraction,
n_per_category, seed). The sidecar is purely diagnostic — `build_examples`
will read it back on a subsequent call to skip re-selection.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from paper2data.llm_summaries import (
    DEFAULT_PROMPT_VERSION,
    DEFAULT_SUMMARY_PROMPT,
    summarize_corpus,
)

log = logging.getLogger(__name__)


@dataclass
class FewShotExample:
    train_idx: int  # index into the X_train pool — for reproducibility
    label_idx: int
    label_name: str
    summary: str


def select_example_indices(
    y_train: np.ndarray, *, n_per_category: int, seed: int
) -> list[int]:
    """Deterministically pick `n_per_category` indices per class from y_train.

    Stable across runs for the same (y_train, n_per_category, seed). Output is
    sorted by (label_idx, original_position) so the rendered prompt has a
    predictable class ordering.
    """
    rng = np.random.default_rng(seed)
    picks: list[int] = []
    for label in np.unique(y_train):
        candidates = np.flatnonzero(y_train == label)
        if len(candidates) < n_per_category:
            raise ValueError(
                f"Class {label} has only {len(candidates)} training docs "
                f"but n_per_category={n_per_category}."
            )
        chosen = rng.choice(candidates, size=n_per_category, replace=False)
        picks.extend(sorted(int(i) for i in chosen))
    return picks


def _sidecar_path(
    cache_dir: Path, *, model_tag: str, prompt_version: str,
    fraction: float, n_per_category: int, seed: int,
) -> Path:
    safe_tag = model_tag.replace("/", "_").replace(":", "_")
    name = (
        f"{safe_tag}__{prompt_version}__frac{fraction}"
        f"__n{n_per_category}__seed{seed}.json"
    )
    return cache_dir / name


def build_examples(
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_names: list[str],
    *,
    n_per_category: int,
    fraction: float,
    model_tag: str,
    summary_cache_dir: str | Path,
    fewshot_cache_dir: str | Path,
    num_ctx: int = 16384,
    temperature: float = 0.0,
    seed: int = 0,
    prompt_template: str = DEFAULT_SUMMARY_PROMPT,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    client=None,
) -> list[FewShotExample]:
    """Pick, summarize, and persist few-shot examples.

    Reads the sidecar JSON if it already exists and returns immediately —
    summarization still respects its own cache, but skipping the load+parse
    keeps the hot path cheap and makes the example set obviously
    reproducible from inspection.
    """
    fewshot_cache_dir = Path(fewshot_cache_dir)
    fewshot_cache_dir.mkdir(parents=True, exist_ok=True)
    sidecar = _sidecar_path(
        fewshot_cache_dir,
        model_tag=model_tag, prompt_version=prompt_version,
        fraction=float(fraction), n_per_category=n_per_category, seed=seed,
    )

    if sidecar.exists():
        log.info("few_shot: reusing sidecar %s", sidecar.name)
        raw = json.loads(sidecar.read_text())
        return [FewShotExample(**r) for r in raw]

    indices = select_example_indices(
        y_train, n_per_category=n_per_category, seed=seed,
    )
    docs = [X_train[i] for i in indices]
    log.info(
        "few_shot: summarizing %d examples (%d per class) at fraction=%.2f",
        len(docs), n_per_category, fraction,
    )
    summaries = summarize_corpus(
        docs,
        fraction=float(fraction),
        model_tag=model_tag,
        cache_dir=summary_cache_dir,
        num_ctx=num_ctx,
        temperature=temperature,
        seed=seed,
        show_progress=False,
        prompt_template=prompt_template,
        prompt_version=prompt_version,
    )

    examples = [
        FewShotExample(
            train_idx=int(idx),
            label_idx=int(y_train[idx]),
            label_name=target_names[int(y_train[idx])],
            summary=s.text,
        )
        for idx, s in zip(indices, summaries)
    ]

    sidecar.write_text(json.dumps([asdict(e) for e in examples], indent=2))
    log.info("few_shot: wrote sidecar %s", sidecar.name)
    return examples


def render_examples_block(examples: list[FewShotExample]) -> str:
    """Render examples as POST/RESPONSE-JSON pairs.

    The RESPONSE shape mirrors the JSON schema the classifier is forced to
    emit (`{"category": "<name>"}`), so the model sees worked examples in
    exactly the form it will be sampled into.
    """
    blocks = []
    for e in examples:
        response = json.dumps({"category": e.label_name})
        blocks.append(f"POST:\n{e.summary}\nRESPONSE: {response}")
    return "\n\n".join(blocks) + "\n"
