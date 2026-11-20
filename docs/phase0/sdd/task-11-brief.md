### Task 11: End-to-end smoke run on a real training split

**Files:** Create `tor/probes/smoke.py`, `tests/test_smoke.py`

**Interfaces:**
- Consumes: `load_metadata`, `load_train` (Task 5); `support_draw` (Task 6); registry and `train_epochs` (Task 7); `acquire_gpu`, `measure`, `append`.
- Produces: `choose_smoke_dataset(meta) -> str`, a seeded rule: candidates are UCR rows with `equal_length & ~missing & n_train <= 500 & length <= 512 & classes <= 10`, sorted by name, and `numpy.random.default_rng(20261120).choice` picks one. Also `run_smoke(adapter_name, mode, epochs=5, device: str | None = None) -> RunRecord` (`kind="smoke"`, full training split, budget `("fraction",1.0)`; `extra` has `losses` and `dataset`; status `ok` requires finite losses and a last loss below the first).
- [ ] **Step 1: Write failing tests**:
  - `test_smoke_choice_is_deterministic_and_eligible`: on the committed `data/metadata.csv`, two calls return the same name, and that row meets every filter.
  - `test_smoke_never_reads_test`: with `load_test` monkeypatched to raise, `run_smoke("inceptiontime","scratch",epochs=1)` on CPU succeeds.
- [ ] **Step 2: Run tests.** Expected: FAIL.
- [ ] **Step 3: Implement**.
- [ ] **Step 4: Run tests.** Expected: 2 passed. Then run the smoke test for every available adapter × supported mode. Record the chosen dataset name in the report as **declared pilot-exposed**.
- [ ] **Step 5: Commit** `"Add smoke run and records"`.

