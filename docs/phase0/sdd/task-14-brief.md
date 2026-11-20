### Task 14: Phase 0 report

**Files:** Create `tor/report.py`, `tests/test_report.py`, `reports/phase0.md` (generated)

**Interfaces:**
- Consumes: `read_all` (Task 2), the availability CSV (Task 8), `FITS_SHAPE` (Task 10), and the markdown files from Task 13.
- Produces: `model_table(records, availability) -> DataFrame` with columns `model, family, available, licence, params_total, fits_full, fits_lora, fits_head, ckpt_needed, s_per_epoch_fits (full), shared_flag, hosted, notes`. `fits_*` is `yes`, `no (oom)`, or `missing: <reason>`. `s_per_epoch` comes only from the `FITS_SHAPE` records; when both shared and unshared records exist, the unshared one is preferred, and the flag column shows which was used. Also `render_report() -> str`.
- [ ] **Step 1: Write failing tests**:
  - `test_missing_shown_not_imputed`: a model with no `FITS_SHAPE` record gives `fits_full == "missing: no probe record"`.
  - `test_unshared_preferred`: two records, one shared at 10 s and one unshared at 8 s, give 8 with `shared_flag False`.
  - `test_report_has_no_score_columns`: the rendered markdown doesn't contain "accuracy", "F1" or "val_".
- [ ] **Step 2: Run tests.** Expected: FAIL.
- [ ] **Step 3: Implement**. The report sections follow the brief's format: what was run, the model table, the full probe grid as an appendix CSV link, the training-free timing, the TabPFN limits, HC2 availability, the literature, deviations (including the `TOR_CACHE` location change and any `not_run`), the declared pilot-exposed smoke dataset, anything unexpected, and next steps.
- [ ] **Step 4: Run tests**, then `python -m tor.report > reports/phase0.md`. Read it end to end and check that every number traces to a run record.
- [ ] **Step 5: Commit** `"Add Phase 0 report"`.
