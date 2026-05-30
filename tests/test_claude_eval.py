"""Tests for the incremental Claude-eval store + sampling. No network.

Run: `uv run python -m unittest tests.test_claude_eval -v`.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from paper2data.claude_eval import (
    _append_store,
    load_store,
    select_test_indices,
)


class SelectTestIndices(unittest.TestCase):
    def setUp(self):
        # 5 classes, 50 docs each, shuffled.
        rng = np.random.default_rng(0)
        self.y = rng.permutation(np.repeat(np.arange(5), 50))

    def test_counts_per_class(self):
        idx = select_test_indices(self.y, 3, seed=42)
        self.assertEqual(len(idx), 15)
        labels = self.y[idx]
        for c in range(5):
            self.assertEqual((labels == c).sum(), 3)

    def test_prefix_stable_growing_only_appends(self):
        small = set(select_test_indices(self.y, 2, seed=42))
        big = set(select_test_indices(self.y, 5, seed=42))
        # Everything chosen at n=2 must still be chosen at n=5.
        self.assertTrue(small.issubset(big))
        self.assertEqual(len(big - small), 5 * 3)  # +3 per class

    def test_deterministic(self):
        self.assertEqual(select_test_indices(self.y, 4, seed=7),
                         select_test_indices(self.y, 4, seed=7))

    def test_raises_when_too_few(self):
        with self.assertRaises(ValueError):
            select_test_indices(self.y, 999, seed=42)


class StoreDedup(unittest.TestCase):
    def test_append_and_dedup_logic(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictions.jsonl"
            self.assertEqual(load_store(path), [])

            y = np.repeat(np.arange(3), 10)
            sel1 = select_test_indices(y, 1, seed=1)
            _append_store(path, [{"test_idx": i} for i in sel1])

            store = load_store(path)
            done = {r["test_idx"] for r in store}
            self.assertEqual(done, set(sel1))

            # Grow to 2/class: only the genuinely new indices should be processed.
            sel2 = select_test_indices(y, 2, seed=1)
            new = [i for i in sel2 if i not in done]
            self.assertEqual(set(new), set(sel2) - set(sel1))
            self.assertEqual(len(new), 3)  # +1 per class, no overlap

            _append_store(path, [{"test_idx": i} for i in new])
            self.assertEqual(len(load_store(path)), len(sel2))


class SummarizeBatchDuplicates(unittest.TestCase):
    """20NG has reposts; identical docs must not produce duplicate batch ids."""

    def test_duplicate_docs_collapse_to_unique_custom_ids(self):
        from unittest import mock

        from paper2data.llm_summaries import summarize_corpus

        class FakeBatch:
            name = "fake"
            concurrency = 1
            seen_ids: list[str] = []

            def supports_batch(self):
                return True

            def generate_batch(self, items, *, model, options):
                ids = [it.custom_id for it in items]
                FakeBatch.seen_ids = ids
                # The API rejects duplicate custom_ids; assert we never send them.
                assert len(ids) == len(set(ids)), f"duplicate custom_ids: {ids}"
                # Echo the full prompt so distinct docs get distinct summaries.
                return {it.custom_id: it.prompt for it in items}

        docs = ["alpha beta gamma delta", "zeta eta theta iota",
                "alpha beta gamma delta", "unique lone document",
                "zeta eta theta iota"]
        fp = FakeBatch()
        with tempfile.TemporaryDirectory() as d:
            res = summarize_corpus(docs, fraction=0.25, model_tag="claude-sonnet-4-6",
                                   cache_dir=d, client=fp, show_progress=False)

        self.assertEqual(len(fp.seen_ids), 3)        # 3 distinct docs
        self.assertEqual(len(res), 5)                 # but 5 results returned
        self.assertTrue(all(r.text for r in res))
        self.assertEqual(res[0].text, res[2].text)    # duplicates share a summary
        self.assertEqual(res[1].text, res[4].text)
        self.assertNotEqual(res[0].text, res[1].text)


if __name__ == "__main__":
    unittest.main()
