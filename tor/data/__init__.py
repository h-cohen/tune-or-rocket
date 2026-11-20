"""Data access: published metadata, train-only loading, Phase 0 test guard."""
from tor.data.load import PhaseGuardError, load_test, load_train
from tor.data.metadata import (
    build_metadata,
    length_percentiles,
    load_metadata,
    representative_channels,
)

__all__ = [
    "PhaseGuardError",
    "build_metadata",
    "length_percentiles",
    "load_metadata",
    "load_test",
    "load_train",
    "representative_channels",
]
