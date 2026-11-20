"""Dataset metadata built from published summaries only (never from data files).

Sources:
- aeon.datasets.get_dataset_meta_data(): downloads the published CSV
  https://timeseriesclassification.com/aeon-toolkit/metadata.csv
  (TrainSize, TestSize, Length, NumberClasses, Channels).
- aeon.datasets.tsc_datasets name lists (hard-coded published archive membership):
  `univariate` (UCR, 128) and `multivariate` (UEA, 30), plus the
  `*_unequal_length`/`*_variable_length` and `univariate_missing_values` lists.

Fields unavailable for a dataset are NaN; nothing is computed from train/test files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tor.config import get_settings

SOURCE_URL = "https://timeseriesclassification.com/aeon-toolkit/metadata.csv"
# Archive name -> name used in the published CSV (same dataset: 500/2425, length 1024).
CSV_ALIASES = {"MixedShapesRegularTrain": "MixedShapes"}
COLUMNS = [
    "name", "archive", "n_train", "n_test", "length", "channels", "classes",
    "equal_length", "missing", "source_url",
]


def build_metadata() -> pd.DataFrame:
    from aeon.datasets import get_dataset_meta_data
    from aeon.datasets import tsc_datasets as t

    pub = get_dataset_meta_data(url=SOURCE_URL).set_index("Dataset")
    unequal = set(t.univariate_variable_length) | set(t.multivariate_unequal_length)
    missing = set(t.univariate_missing_values)
    rows = []
    for archive, names in (("UCR", t.univariate), ("UEA", t.multivariate)):
        for name in sorted(names):
            key = CSV_ALIASES.get(name, name)
            r = pub.loc[key] if key in pub.index else None
            eq = name not in unequal and (r is None or int(r["Length"]) != 0)
            src = SOURCE_URL + (f"#{key}" if key != name else "")
            rows.append({
                "name": name,
                "archive": archive,
                "n_train": np.nan if r is None else r["TrainSize"],
                "n_test": np.nan if r is None else r["TestSize"],
                "length": np.nan if (r is None or not eq) else int(r["Length"]),
                "channels": np.nan if r is None else r["Channels"],
                "classes": np.nan if r is None else r["NumberClasses"],
                "equal_length": eq,
                "missing": name in missing,
                "source_url": src if r is not None else f"{SOURCE_URL}#absent",
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def metadata_path() -> Path:
    return get_settings().repo_root / "data" / "metadata.csv"


def load_metadata() -> pd.DataFrame:
    return pd.read_csv(metadata_path())


def length_percentiles(meta: pd.DataFrame) -> dict:
    """p10/p50/p90/max of `length` over rows with equal_length & ~missing.

    Uses numpy.percentile with default linear interpolation, then int() (truncation).
    """
    sel = meta[meta["equal_length"].astype(bool) & ~meta["missing"].astype(bool)]
    L = sel["length"].dropna().to_numpy(dtype=float)
    p10, p50, p90 = (int(np.percentile(L, q)) for q in (10, 50, 90))
    return {"p10": p10, "p50": p50, "p90": p90, "max": int(L.max())}


def representative_channels(meta: pd.DataFrame) -> int:
    """Median `channels` over UEA rows with channels > 1, rounded."""
    c = meta[(meta["archive"] == "UEA") & (meta["channels"] > 1)]["channels"]
    return int(round(float(c.median())))


if __name__ == "__main__":
    df = build_metadata()
    p = metadata_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"wrote {p} ({len(df)} rows)")
