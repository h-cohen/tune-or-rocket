"""Probe shapes: (length, channels, batch) grid derived from published metadata only."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tor.data.metadata import length_percentiles, representative_channels

BATCHES = (8, 32, 64)


@dataclass(frozen=True)
class ProbeShape:
    length: int
    channels: int
    batch: int


def probe_grid(meta: pd.DataFrame) -> list[ProbeShape]:
    """4 length percentiles x channels {1, representative} x batch {8, 32, 64}, deduplicated.

    Lengths come from equal-length datasets (missing values allowed: they do not
    change compute shape).
    """
    pct = length_percentiles(meta, require_no_missing=False)
    chans = sorted({1, representative_channels(meta)})
    out: list[ProbeShape] = []
    for length in sorted({pct["p10"], pct["p50"], pct["p90"], pct["max"]}):
        for c in chans:
            for b in BATCHES:
                s = ProbeShape(length, c, b)
                if s not in out:
                    out.append(s)
    return out


def FITS_SHAPE(meta: pd.DataFrame) -> ProbeShape:  # noqa: N802 - name fixed by the brief
    """p90 length, 1 channel, batch 8: the shape used to test whether a model fits."""
    return ProbeShape(length_percentiles(meta, require_no_missing=False)["p90"], 1, 8)
