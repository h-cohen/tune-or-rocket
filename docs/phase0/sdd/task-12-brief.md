### Task 12: Training-free arms and TabPFN limits

**Files:** Create `tor/baselines.py`, `tor/tabpfn_arms.py`, `tests/test_baselines.py`

**Interfaces:**
- Consumes: `probe_grid` (Task 10), `measure`, `append`.
- Produces: `TRAINING_FREE = {"minirocket": ..., "mr_hydra": ..., "quant": ...}`, each a factory returning an aeon/sklearn pipeline (MiniRocket + `RidgeClassifierCV(alphas=np.logspace(-3,3,10))`; MultiRocket and Hydra features concatenated with the same ridge; QUANT + `ExtraTreesClassifier(n_estimators=200, max_features=0.1, criterion="entropy")` as in aeon's QUANT defaults, to be verified). `time_baseline(name, shape: ProbeShape, n_series=512) -> RunRecord` (fit + transform time on synthetic data, CPU only, `n_jobs=16`). `tabpfn_limits() -> dict` (version, max samples/features/classes per the installed package and docs, whether a GPU is used, local vs hosted). `rocketpfn_pipeline()` and `masht_pipeline()`, built from each paper's description of its feature reduction or ensembling step, cited in docstrings; any detail the paper doesn't give is recorded in `extra["unspecified"]`. `time_tabpfn(recipe: Literal["rocketpfn","masht"], shape) -> RunRecord` (hosted runs use `tabpfn-client`, read `TABPFN_TOKEN`, and set `hosted=True`). `datasets_exceeding_limits(meta, limits) -> list[str]`.
- [ ] **Step 1: Write failing tests**:
  - `test_baselines_fit_on_synthetic`: each of the 3 factories fits and predicts on `randn(40,1,64)` with 2 classes without error (prediction output is used only for its shape).
  - `test_exceeding_limits`: a fixture where one row has `classes=12` and `limits["max_classes"]=10` returns that row's name.
  - `test_hosted_run_flags_and_needs_token`: with `TABPFN_TOKEN` unset, `time_tabpfn("masht", ...)` with hosted mode returns `status=="not_run: no TabPFN token"`. With the client mocked, `cost["hosted"] is True`.
- [ ] **Step 2: Run tests.** Expected: FAIL.
- [ ] **Step 3: Implement**. Install `tabpfn` and `tabpfn-client`, then refresh the lock file.
- [ ] **Step 4: Run tests**, then time every baseline and both TabPFN recipes on the `FITS_SHAPE` and p50 shapes. Write the `tabpfn_limits` output as a `kind="tabpfn_limits"` record.
- [ ] **Step 5: Commit** `"Add training-free arms, TabPFN recipes and limits"`.

