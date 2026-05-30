"""Pluggable LLM backends for the summarize + classify pipeline.

The rest of the codebase is written against the Ollama client's duck-typed
shape:

    client.generate(model=..., prompt=..., options={...})  -> {"response": text}
    client.chat(model=..., messages=[...], format=schema, options={...})
        -> {"message": {"content": json_str}}

To add Claude without rewriting the pipeline, each backend here implements that
same shape. `make_provider(cfg.llm)` picks one based on a `provider:` field:

    "ollama"     -> OllamaProvider     (qwen2.5 etc., unchanged)
    "claude_api" -> ClaudeApiProvider  (anthropic SDK, Messages API + Batch)

Two extra hooks let the corpus-level loops exploit a backend's strengths
without changing the per-doc functions:

    .concurrency        how many docs to run in parallel (1 = serial)
    .supports_batch()   whether to collect cache-misses and submit one batch

Claude ignores `num_ctx`/`seed` (it manages context itself and has no seed); we
keep accepting them so cache keys and call sites stay identical to the Ollama
path. `max_tokens` is passed out-of-band (Ollama ignores it to preserve its
existing behavior; the API backend requires it).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Sonnet 4.6 list price, used only to estimate cost from token usage when the
# backend does not report dollars directly (the API/batch path).
_PRICE_PER_MTOK = {"input": 3.0, "output": 15.0, "cache_read": 0.30}


@dataclass
class BatchItem:
    """One unit of work for a batch submission.

    `custom_id` is the per-doc cache key, so results map straight back onto the
    on-disk cache. Summaries set `prompt`; classification sets `messages`.
    """

    custom_id: str
    prompt: str | None = None
    messages: list[dict] | None = None
    max_tokens: int = 1024


@dataclass
class _Usage:
    """Thread-safe running tally of tokens / dollars across calls."""

    n_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    total_cost_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, *, input_tokens=0, output_tokens=0, cache_read=0,
            cache_creation=0, cost_usd=0.0) -> None:
        with self._lock:
            self.n_calls += 1
            self.input_tokens += int(input_tokens or 0)
            self.output_tokens += int(output_tokens or 0)
            self.cache_read_input_tokens += int(cache_read or 0)
            self.cache_creation_input_tokens += int(cache_creation or 0)
            self.total_cost_usd += float(cost_usd or 0.0)

    def as_dict(self) -> dict:
        with self._lock:
            return {
                "n_calls": self.n_calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cache_read_input_tokens": self.cache_read_input_tokens,
                "cache_creation_input_tokens": self.cache_creation_input_tokens,
                "total_cost_usd": round(self.total_cost_usd, 6),
            }


# --------------------------------------------------------------------------- #
# Ollama (unchanged behaviour)
# --------------------------------------------------------------------------- #
class OllamaProvider:
    """Thin wrapper around `ollama.Client` so call sites receive a provider.

    Behaviour is byte-for-byte identical to calling the client directly; this
    keeps existing qwen2.5 runs and their cache entries unchanged.
    """

    name = "ollama"
    concurrency = 1

    def __init__(self, client):
        self._client = client

    def supports_batch(self) -> bool:
        return False

    def generate(self, *, model, prompt, options, max_tokens=None) -> dict:
        return self._client.generate(model=model, prompt=prompt, options=options)

    def chat(self, *, model, messages, format, options, max_tokens=None) -> dict:
        return self._client.chat(
            model=model, messages=messages, format=format, options=options
        )

    def usage_summary(self) -> dict:
        return {}


# --------------------------------------------------------------------------- #
# Claude Messages API + Batch (anthropic SDK) — pay per token
# --------------------------------------------------------------------------- #
class ClaudeApiProvider:
    """Anthropic Messages API backend with optional Batch API (50% off, async).

    Classification uses forced tool use: a single tool whose input_schema is the
    pydantic Classification schema, so the model is constrained to emit
    {"category": ...}. Summaries are plain text. When `use_batch` is set, the
    corpus loops collect all cache-misses and submit them as one batch.
    """

    name = "claude_api"

    def __init__(self, *, client=None, model=None, use_batch=True,
                 prompt_caching=False, poll_interval_s=30, concurrency=4,
                 batch_dir=None, max_wait_s=24 * 3600):
        if client is None:
            try:
                import anthropic
            except ImportError as e:
                raise RuntimeError(
                    "anthropic SDK not installed; `uv add anthropic` to use "
                    "provider=claude_api."
                ) from e
            client = anthropic.Anthropic()
        self._client = client
        self._model = model
        self._use_batch = bool(use_batch)
        self._prompt_caching = bool(prompt_caching)  # reserved for follow-up
        self._poll_interval_s = int(poll_interval_s)
        self._max_wait_s = int(max_wait_s)
        self.concurrency = int(concurrency)
        self._batch_dir = batch_dir
        self._usage = _Usage()
        self.batch_ids: list[str] = []

    def supports_batch(self) -> bool:
        return self._use_batch

    # -- usage / cost ------------------------------------------------------- #
    def _record(self, usage) -> None:
        if usage is None:
            return
        inp = getattr(usage, "input_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        cr = getattr(usage, "cache_read_input_tokens", 0) or 0
        cc = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cost = (
            inp / 1e6 * _PRICE_PER_MTOK["input"]
            + out / 1e6 * _PRICE_PER_MTOK["output"]
            + cr / 1e6 * _PRICE_PER_MTOK["cache_read"]
        )
        if self._use_batch:
            cost *= 0.5  # Batch API discount
        self._usage.add(input_tokens=inp, output_tokens=out, cache_read=cr,
                        cache_creation=cc, cost_usd=cost)

    # -- single (non-batch) calls ------------------------------------------ #
    def generate(self, *, model, prompt, options, max_tokens=None) -> dict:
        resp = self._client.messages.create(
            model=model or self._model,
            max_tokens=int(max_tokens or 1024),
            temperature=float(options.get("temperature", 0.0)),
            messages=[{"role": "user", "content": prompt}],
        )
        self._record(getattr(resp, "usage", None))
        return {"response": _first_text(resp)}

    def chat(self, *, model, messages, format, options, max_tokens=None) -> dict:
        tool = _classify_tool(format)
        resp = self._client.messages.create(
            model=model or self._model,
            max_tokens=int(max_tokens or 64),
            temperature=float(options.get("temperature", 0.0)),
            messages=messages,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
        )
        self._record(getattr(resp, "usage", None))
        return {"message": {"content": json.dumps(_first_tool_input(resp))}}

    # -- batch calls -------------------------------------------------------- #
    def generate_batch(self, items, *, model, options) -> dict[str, str]:
        requests = [
            {
                "custom_id": it.custom_id,
                "params": {
                    "model": model or self._model,
                    "max_tokens": int(it.max_tokens),
                    "temperature": float(options.get("temperature", 0.0)),
                    "messages": [{"role": "user", "content": it.prompt}],
                },
            }
            for it in items
        ]
        return self._run_batch(requests, kind="text")

    def chat_batch(self, items, *, model, format, options) -> dict[str, str]:
        tool = _classify_tool(format)
        requests = [
            {
                "custom_id": it.custom_id,
                "params": {
                    "model": model or self._model,
                    "max_tokens": int(it.max_tokens),
                    "temperature": float(options.get("temperature", 0.0)),
                    "messages": it.messages,
                    "tools": [tool],
                    "tool_choice": {"type": "tool", "name": tool["name"]},
                },
            }
            for it in items
        ]
        return self._run_batch(requests, kind="tool")

    def _run_batch(self, requests, *, kind) -> dict[str, str]:
        batch = self._client.messages.batches.create(requests=requests)
        self.batch_ids.append(batch.id)
        log.info("claude_api: submitted batch %s with %d requests",
                 batch.id, len(requests))

        deadline = time.time() + self._max_wait_s
        while True:
            batch = self._client.messages.batches.retrieve(batch.id)
            if batch.processing_status == "ended":
                break
            if time.time() > deadline:
                raise TimeoutError(
                    f"batch {batch.id} not done after {self._max_wait_s}s "
                    f"(status={batch.processing_status})"
                )
            log.info("claude_api: batch %s status=%s counts=%s; sleeping %ds",
                     batch.id, batch.processing_status,
                     batch.request_counts, self._poll_interval_s)
            time.sleep(self._poll_interval_s)

        out: dict[str, str] = {}
        for entry in self._client.messages.batches.results(batch.id):
            cid = entry.custom_id
            result = entry.result
            if result.type != "succeeded":
                log.warning("claude_api: %s -> %s (skipped)", cid, result.type)
                continue
            msg = result.message
            self._record(getattr(msg, "usage", None))
            if kind == "text":
                out[cid] = _first_text(msg)
            else:
                out[cid] = json.dumps(_first_tool_input(msg))
        return out

    def usage_summary(self) -> dict:
        d = self._usage.as_dict()
        d["batch_ids"] = list(self.batch_ids)
        return d


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _first_text(message) -> str:
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "text":
            return block.text.strip()
    return ""


def _first_tool_input(message) -> dict:
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "tool_use":
            return dict(block.input)
    return {}


def _classify_tool(schema: dict, *, name: str = "classify") -> dict:
    """Wrap the pydantic JSON schema as an Anthropic tool input_schema.

    Strips `title` keys and forces additionalProperties=false, which the
    structured-output path requires; Ollama accepts the raw schema.
    """
    clean = _strip_titles(schema)
    clean["additionalProperties"] = False
    clean.setdefault("type", "object")
    return {
        "name": name,
        "description": "Return the single best category for the post.",
        "input_schema": clean,
    }


def _strip_titles(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_titles(v) for k, v in obj.items() if k != "title"}
    if isinstance(obj, list):
        return [_strip_titles(v) for v in obj]
    return obj


def make_provider(cfg_llm):
    """Build a provider from an llm config node (DictConfig or dataclass)."""
    provider = str(getattr(cfg_llm, "provider", "ollama"))

    if provider == "ollama":
        import ollama
        return OllamaProvider(ollama.Client(host=os.environ.get("OLLAMA_HOST")))

    if provider == "claude_api":
        return ClaudeApiProvider(
            model=str(cfg_llm.tag),
            use_batch=bool(getattr(cfg_llm, "use_batch", True)),
            prompt_caching=bool(getattr(cfg_llm, "prompt_caching", False)),
            poll_interval_s=int(getattr(cfg_llm, "poll_interval_s", 30)),
            concurrency=int(getattr(cfg_llm, "max_concurrency", 4)),
        )

    raise ValueError(f"unknown llm.provider {provider!r}")
