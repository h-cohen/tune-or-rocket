"""Availability of published full-label HC2 / MR-Hydra results (Phase 0: availability only).

Only dataset names and the count of non-missing resample columns are read from the
published CSVs; no accuracy value is used, aggregated or reported.

Sources (first source that contains a dataset wins; recorded in `source_url`):
- aeon `aeon.benchmarking.results_loaders.get_estimator_results` default path
  https://timeseriesclassification.com/results/ReferenceResults/classification/accuracy/
  `<name>_accuracy.csv` (UCR; aeon docstring: "The first resample is the default
  train/test split").
- HC2 only, UEA: https://timeseriesclassification.com/results/PublishedResults/HIVE-COTEV2/HC2_TESTFOLDS.csv
"""
from __future__ import annotations

from typing import Callable, Literal

import pandas as pd

REF = "https://timeseriesclassification.com/results/ReferenceResults/classification/accuracy"
PUB = "https://timeseriesclassification.com/results/PublishedResults"
SOURCES: dict[str, list[str]] = {
    "HC2": [f"{REF}/HC2_accuracy.csv", f"{PUB}/HIVE-COTEV2/HC2_TESTFOLDS.csv"],
    "MR-Hydra": [f"{REF}/MR-Hydra_accuracy.csv"],
}
COLUMNS = ["name", "available", "resamples", "source_url"]


def _read_names(url: str, fetch: Callable[[str], pd.DataFrame]) -> dict[str, int]:
    """dataset name -> number of non-NaN resample columns (values are never kept)."""
    df = fetch(url)
    counts = df.iloc[:, 1:].notna().sum(axis=1)
    return {str(n): int(c) for n, c in zip(df.iloc[:, 0], counts)}


def published_results_coverage(
    meta: pd.DataFrame,
    classifier: Literal["HC2", "MR-Hydra"],
    sources: list[str] | None = None,
    fetch: Callable[[str], pd.DataFrame] = pd.read_csv,
) -> pd.DataFrame:
    """One row per `meta` dataset: name, available, resamples (int or None), source_url.

    Names are matched exactly (no aliasing); datasets absent from every source are
    reported unavailable with source_url None.
    """
    urls = SOURCES[classifier] if sources is None else sources
    tables = [(u, _read_names(u, fetch)) for u in urls]
    rows = []
    for name in meta["name"]:
        hit = next(((u, t[name]) for u, t in tables if name in t), None)
        rows.append({
            "name": name,
            "available": hit is not None,
            "resamples": hit[1] if hit else None,
            "source_url": hit[0] if hit else None,
        })
    out = pd.DataFrame(rows, columns=COLUMNS)
    out["resamples"] = out["resamples"].astype("Int64")
    return out


if __name__ == "__main__":
    from tor.data.metadata import load_metadata

    m = load_metadata()
    for clf in SOURCES:
        c = published_results_coverage(m, clf).merge(m[["name", "archive"]], on="name")
        print(clf, c.groupby("archive")["available"].agg(["sum", "count"]).to_dict("index"))
