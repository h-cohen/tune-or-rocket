"""Seeded support draws from the TRAINING split only (never reads test data)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np

from tor.config import get_settings

Budget = tuple[Literal["per_class", "fraction"], float]

BUDGETS: list[Budget] = [
    ("per_class", 5),
    ("per_class", 20),
    ("fraction", 0.1),
    ("fraction", 1.0),
]
DRAWS = range(10)


class InsufficientClassError(ValueError):
    """A class has fewer training examples than the per-class budget."""


def draw_seed(dataset: str, budget: Budget, draw: int) -> int:
    """Seed = int.from_bytes(sha256(s.encode()).digest()[:8], "big") with
    s = f"{dataset}|{kind}|{value}|{draw}", where value is str(value) of the
    budget value exactly as listed in BUDGETS (e.g. "5", "20", "0.1", "1.0").
    This string format must NEVER change after Phase 0.
    """
    kind, value = budget
    s = f"{dataset}|{kind}|{value}|{draw}"
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big")


def support_draw(y: np.ndarray, budget: Budget, draw: int, dataset: str) -> np.ndarray:
    """Sorted train indices for one draw, sampled without replacement per class.

    per_class k: k per class; raises InsufficientClassError if any class has < k.
    fraction f: per class max(1, round(f * n_class)) using Python round
    (banker's rounding); f == 1.0 returns arange(len(y)) exactly.
    Classes are iterated in sorted label order so results do not depend on the
    order of labels in y.
    """
    y = np.asarray(y)
    kind, value = budget
    if kind == "fraction" and value == 1.0:
        return np.arange(len(y))
    rng = np.random.default_rng(draw_seed(dataset, budget, draw))
    picked = []
    for label in sorted(np.unique(y).tolist()):
        members = np.flatnonzero(y == label)
        n = len(members)
        if kind == "per_class":
            k = int(value)
            if n < k:
                raise InsufficientClassError(
                    f"class {label!r} has {n} training examples, need {k} (budget per_class={k})"
                )
        elif kind == "fraction":
            k = min(n, max(1, round(value * n)))
        else:
            raise ValueError(f"unknown budget kind {kind!r}")
        picked.append(rng.choice(members, size=k, replace=False))
    return np.sort(np.concatenate(picked))


def budget_key(budget: Budget) -> str:
    return f"{budget[0]}={budget[1]}"


def save_draws(dataset: str, y: np.ndarray) -> Path:
    """Write splits/{dataset}.json: {budget_key: {draw: [indices]}}.
    A budget raising InsufficientClassError is stored as {"error": message}."""
    out: dict[str, dict] = {}
    for b in BUDGETS:
        try:
            out[budget_key(b)] = {
                str(d): support_draw(y, b, d, dataset).tolist() for d in DRAWS
            }
        except InsufficientClassError as e:
            out[budget_key(b)] = {"error": str(e)}
    path = get_settings().repo_root / "splits" / f"{dataset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return path
