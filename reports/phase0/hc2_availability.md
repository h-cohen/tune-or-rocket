# HC2 and MR-Hydra: published full-label results (availability only)

Code: `tor/hc2.py` (`published_results_coverage`). Only dataset names and the number of
non-missing resample columns were read from the published CSVs. No accuracy value was
used, aggregated or reported here. Checked 2026-11-20 against our `data/metadata.csv`
(128 UCR, 30 UEA rows).

## Sources (verified by download)

- UCR, HC2 and MR-Hydra: aeon 1.6.0 `aeon.benchmarking.results_loaders.get_estimator_results`
  (default `path`) reads
  `https://timeseriesclassification.com/results/ReferenceResults/classification/accuracy/HC2_accuracy.csv`
  and `.../MR-Hydra_accuracy.csv`. aeon maps "MultiRocketHydra", "Hydra-MR" etc. to `MR-Hydra`.
- UEA, HC2 only: `https://timeseriesclassification.com/results/PublishedResults/HIVE-COTEV2/HC2_TESTFOLDS.csv`
  (138 datasets: 112 UCR + 26 UEA).
- UEA, MR-Hydra: no published file found. The reference-results directory holds only
  the 112 UCR datasets; `PublishedResults/Bakeoff2021` (multivariate bake-off) has no
  HC2 or Hydra files; `Bakeoff2023` (UCR) has `Hydra-MR_TESTFOLDS.csv` with 0 UEA names.
  Not searched: the tsml-eval GitHub results directories, so an MR-Hydra UEA file may
  exist there (unverified).
- aeon 1.6.0 has no loader for the UEA HC2 file; it was read directly via URL.

## Coverage (datasets with a published result / datasets in our metadata)

| Archive | HC2 | MR-Hydra |
|---|---|---|
| UCR | 112 / 128 | 112 / 128 |
| UEA | 26 / 30 | 0 / 30 |

Every available dataset has 30 resamples (columns 0 to 29) in every file.

## Datasets without published results

- UCR, HC2 and MR-Hydra (same 16 datasets): AllGestureWiimoteX, AllGestureWiimoteY,
  AllGestureWiimoteZ, DodgerLoopDay, DodgerLoopGame, DodgerLoopWeekend, Fungi,
  GestureMidAirD1, GestureMidAirD2, GestureMidAirD3, GesturePebbleZ1, GesturePebbleZ2,
  MelbournePedestrian, PLAID, PickupGestureWiimoteZ, ShakeGestureWiimoteZ.
  15 of these 16 are variable-length or missing-value datasets that the 112-dataset
  bake-off set excludes. Fungi is the exception: in our metadata it is equal-length with
  no missing values (18 train cases, 18 classes), so it is excluded for a reason not
  established here (one training case per class may be the cause; unverified).
  `Bakeoff2023` lists `...Eq` variants (e.g. `AllGestureWiimoteXEq`),
  which are different, equal-length-processed datasets. We do not alias them.
- UEA, HC2 (4): CharacterTrajectories, InsectWingbeat, JapaneseVowels, SpokenArabicDigits.
- UEA, MR-Hydra: all 30.

## Name mismatches

None. All 112 names in each UCR file and the 26 UEA names in the HC2 file match our
metadata exactly, including `MixedShapesRegularTrain` (the published results use this
name; only the metadata CSV uses `MixedShapes`). No name in any file is absent from our
metadata, so nothing was aliased.

## Resample 0 and the official split

- Verified in installed code: the `get_estimator_results` docstring says "The first
  resample is the default train/test split for the dataset" (`num_resamples=1` returns
  resample 0 only). This is aeon's statement about its loader of these files.
- Primary source for the UCR files: Bake off redux (Middlehurst, Schaefer, Bagnall 2024,
  arXiv 2304.13029), as verified by the reviewer: "Resample 0 uses the original train and
  test split from the UCR archive." and "We resample each pair of train/test data 30 times
  ... stratifying to retain the same class distribution." So resample 0 is the official
  split and resamples 1 to 29 are stratified resamples.
- Not verified: that the UEA file `HIVE-COTEV2/HC2_TESTFOLDS.csv` follows the same
  convention (assumed from tsml practice, not checked).

## Usage restriction

These are full-label results only (the whole official training set, resample 0, or
resamples of the full pool). They can serve as a reference only at the 100% label
budget. They give no information about reduced-label budgets.
