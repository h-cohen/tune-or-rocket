"""Dataset loading. Phase 0 never reads a test split."""
from __future__ import annotations

import numpy as np

from tor.config import get_settings


class PhaseGuardError(Exception):
    """Raised when a test split is requested during Phase 0."""


def _test_allowed(phase: str) -> bool:
    """Fail-closed: allowed only if phase, stripped, is a plain integer >= 1."""
    try:
        return int(phase.strip()) >= 1 and phase.strip().isdigit() and phase.strip() == str(int(phase.strip()))
    except ValueError:
        return False


def load_test(name: str):
    if not _test_allowed(get_settings().phase):  # first statement: no import or I/O before it
        raise PhaseGuardError(
            f"Phase 0: test split of {name!r} must not be read "
            "(TOR_PHASE must be an integer >= 1 to allow it)."
        )
    return _load(name, "test")


def load_train(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the train split only. X has shape (n, c, L); y is an array of str labels.

    There is intentionally no split argument; "train" is always requested explicitly
    (aeon's load_classification with split=None would load both splits).
    """
    return _load(name, "train")


def _load(name: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    from aeon.datasets import load_classification

    path = get_settings().cache_dir / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    X, y = load_classification(name, split=split, extract_path=str(path))
    if isinstance(X, np.ndarray):
        X = np.asarray(X)
    else:
        raise ValueError(f"{name}: unequal-length data not supported by load_train")
    return X, np.asarray(y).astype(str)
