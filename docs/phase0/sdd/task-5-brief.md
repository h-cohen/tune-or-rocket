### Task 5: Data: published metadata, train loading, test guard

**Files:** Create `tor/data/__init__.py`, `tor/data/metadata.py`, `tor/data/load.py`, `tests/test_data.py`, `data/metadata.csv` (generated and committed)

**Interfaces:**
- Produces: `build_metadata() -> pandas.DataFrame` with columns `name, archive (UCR|UEA), n_train, n_test, length (int or NaN if unequal), channels, classes, equal_length: bool, missing: bool, source_url`; `load_metadata() -> DataFrame` (reads `data/metadata.csv`); `length_percentiles(meta) -> dict` with keys `p10, p50, p90, max` (ints, over rows where `equal_length & ~missing`, UCR and UEA together); `representative_channels(meta) -> int` (median `channels` over UEA rows with `channels > 1`, rounded); `load_train(name: str) -> tuple[np.ndarray, np.ndarray]` (X shape `(n, c, L)`, y as str labels); `load_test(name)`; `PhaseGuardError(Exception)`.

- [ ] **Step 1: Write failing tests**:
  - `test_load_test_blocked_in_phase0`: with `TOR_PHASE=0`, `load_test("GunPoint")` raises `PhaseGuardError`, and the message contains `"Phase 0"`.
  - `test_load_test_guard_runs_before_any_io`: monkeypatch `aeon.datasets.load_classification` to raise `AssertionError`. With `TOR_PHASE=0` the call still raises `PhaseGuardError`, which proves the guard runs first.
  - `test_load_train_shape` (marker `network`): `X, y = load_train("GunPoint")` gives `X.shape == (50, 1, 150)` and `len(set(y)) == 2`.
  - `test_length_percentiles_on_fixture`: a 5-row fixture with lengths [10,20,30,40,1000] gives `max == 1000` and `p50 == 30`.
  - `test_metadata_has_no_test_derived_reads`: monkeypatch `load_test` to raise, then call `build_metadata()` (marker `network`). It succeeds.
- [ ] **Step 2: Run** `pytest tests/test_data.py -v -m "not network"`. Expected: FAIL.
- [ ] **Step 3: Implement**. `load_train` calls `aeon.datasets.load_classification(name, split="train", extract_path=TOR_CACHE/"datasets")`. `build_metadata` uses published summaries only. First try aeon's metadata helper (verify the name in the installed aeon; `aeon.datasets` has dataset lists and a metadata accessor in recent versions). Fall back to the timeseriesclassification.com dataset-summary CSV (verify the URL), and record which one was used in `source_url`. If neither gives `n_test` for a dataset, store NaN. Never compute it from files.
- [ ] **Step 4: Run** `pytest tests/test_data.py -v`. Expected: 5 passed. Then `python -m tor.data.metadata` writes `data/metadata.csv`. Check `wc -l data/metadata.csv`: about 128 UCR + 30 UEA rows (report the actual count).
- [ ] **Step 5: Commit** `"Add metadata from published summaries, train loader and test guard"`.

