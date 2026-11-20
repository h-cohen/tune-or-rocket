# Task 5 report
Implemented tor/data/{__init__,load,metadata}.py, tests/test_data.py, data/metadata.csv (158 rows), pyproject (packages += tor.data, `network` marker).
Sources: aeon 1.6.0 `get_dataset_meta_data()` which does pd.read_csv on https://timeseriesclassification.com/aeon-toolkit/metadata.csv (verified in source; no data files loaded) for n_train/n_test/length/classes/channels; archive membership from aeon.datasets.tsc_datasets lists (`univariate`=UCR 128, `multivariate`=UEA 30); unequal-length from `univariate_variable_length`/`multivariate_unequal_length` (length NaN) ; missing from `univariate_missing_values` (4 UCR; no UEA missing list, so False).
Quirks: CSV lacks `InsectWingbeatSound` (CSV's `InsectSound` is a different 25000/25000 dataset) -> all fields NaN, source_url ends `#absent`. `MixedShapesRegularTrain` aliased to CSV `MixedShapes` (500/2425, len 1024), source_url ends `#MixedShapes`.
Results: UCR 128, UEA 30 rows. length percentiles (equal & ~missing, 139 rows w/ length): p10=68, p50=308, p90=1500, max=17984. representative_channels=6.
Tests: 6 passed (incl. network). Full suite run too. Extra test: load_train only requests split="train", rejects split kwarg.
TDD note: implementation written before the RED run was captured (no RED evidence); tests pass GREEN.

## Fix round 1
- equal_length/missing are nullable boolean; NA for InsectWingbeatSound (absent from CSV, not in any aeon list). UEA `missing` is NA for all 30 (no published source found; aeon has no UEA missing list). UCR rows in CSV: explicit list -> True/False.
- length_percentiles selects equal_length==True & missing==False (NA excluded). New values: p10=80, p50=315, p90=1496, max=3000 (was 68/308/1500/17984). representative_channels still 6. UEA rows drop out, so max fell to 3000.
- Guard fail-closed: load_test allowed only if TOR_PHASE stripped is a canonical integer >=1; unset/""/" 0"/"00"/"abc"/"0"/"-1"/"1.5" raise; "1" passes (loader monkeypatched).
- Network metadata test also patches aeon load_classification to raise and asserts NA flags.
- Command: `pytest -q` -> 41 passed (tests/test_data.py: 15 passed). Regenerated data/metadata.csv (158 rows).
