"""Dataset loaders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import fetch_20newsgroups


@dataclass
class TextDataset:
    X: list[str]
    y: np.ndarray
    target_names: list[str]


def load_twenty_newsgroups(
    subset: str = "all",
    remove: tuple[str, ...] = ("headers", "footers", "quotes"),
    categories: list[str] | None = None,
) -> TextDataset:
    bunch = fetch_20newsgroups(
        subset=subset,
        remove=tuple(remove),
        categories=categories,
        shuffle=False,
    )
    return TextDataset(
        X=list(bunch.data),
        y=np.asarray(bunch.target),
        target_names=list(bunch.target_names),
    )
