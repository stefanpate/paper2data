"""Sentence-transformer document embeddings with on-disk caching.

Embeddings are deterministic for (model_id, passage_prefix, normalize, corpus),
so we cache the final dense matrix to disk and reuse it across runs and
across CV folds — encoding is by far the dominant cost.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class EmbedResult:
    vectors: np.ndarray
    cache_key: str
    cache_hit: bool
    elapsed_s: float


def _corpus_fingerprint(texts: list[str]) -> str:
    """Cheap-but-collision-resistant fingerprint of the corpus contents."""
    h = hashlib.sha1()
    h.update(str(len(texts)).encode())
    h.update(b"\0")
    # Hash every doc — fast, and avoids first/last-only collisions if the
    # caller reorders or filters the corpus.
    for t in texts:
        h.update(t.encode("utf-8", errors="replace"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _cache_key(
    *,
    model_id: str,
    passage_prefix: str,
    normalize: bool,
    corpus_fp: str,
) -> str:
    payload = json.dumps(
        {
            "model_id": model_id,
            "passage_prefix": passage_prefix,
            "normalize": normalize,
            "corpus": corpus_fp,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha1(payload).hexdigest()[:24]


def embed_corpus(
    texts: list[str],
    *,
    model_id: str,
    cache_dir: str | Path,
    batch_size: int = 64,
    normalize_embeddings: bool = True,
    passage_prefix: str = "",
    trust_remote_code: bool = False,
    device: str | None = None,
) -> EmbedResult:
    """Embed a corpus, caching the result on disk."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    corpus_fp = _corpus_fingerprint(texts)
    key = _cache_key(
        model_id=model_id,
        passage_prefix=passage_prefix,
        normalize=normalize_embeddings,
        corpus_fp=corpus_fp,
    )
    cache_path = cache_dir / f"{key}.npy"
    meta_path = cache_dir / f"{key}.json"

    if cache_path.exists():
        t0 = time.perf_counter()
        vectors = np.load(cache_path)
        log.info(
            "Embedding cache HIT: %s (%d x %d) in %.2fs",
            cache_path.name, vectors.shape[0], vectors.shape[1], time.perf_counter() - t0,
        )
        return EmbedResult(vectors=vectors, cache_key=key, cache_hit=True,
                           elapsed_s=time.perf_counter() - t0)

    # Import lazily so the rest of the package works without sentence-transformers.
    from sentence_transformers import SentenceTransformer

    log.info("Embedding cache MISS — loading %s", model_id)
    model = SentenceTransformer(model_id, trust_remote_code=trust_remote_code, device=device)
    inputs = [passage_prefix + t for t in texts] if passage_prefix else texts

    t0 = time.perf_counter()
    vectors = model.encode(
        inputs,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    elapsed = time.perf_counter() - t0
    log.info(
        "Encoded %d docs -> %d-dim in %.1fs (%.1f docs/s)",
        vectors.shape[0], vectors.shape[1], elapsed, vectors.shape[0] / max(elapsed, 1e-6),
    )

    np.save(cache_path, vectors)
    meta_path.write_text(json.dumps({
        "model_id": model_id,
        "passage_prefix": passage_prefix,
        "normalize_embeddings": normalize_embeddings,
        "n_docs": int(vectors.shape[0]),
        "dim": int(vectors.shape[1]),
        "corpus_fingerprint": corpus_fp,
    }, indent=2))

    return EmbedResult(vectors=vectors, cache_key=key, cache_hit=False, elapsed_s=elapsed)
