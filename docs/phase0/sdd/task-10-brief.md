### Task 10: Memory and time probe runner

**Files:** Create `tor/probes/__init__.py`, `tor/probes/shapes.py`, `tor/probes/memtime.py`, `tests/test_probes.py`

**Interfaces:**
- Consumes: `load_metadata`, `length_percentiles`, `representative_channels` (Task 5); `REGISTRY`, `train_epochs` (Task 7); `acquire_gpu` (Task 3); `measure` (Task 4); `append` (Task 2).
- Produces: `ProbeShape(length:int, channels:int, batch:int)`; `probe_grid(meta) -> list[ProbeShape]` (lengths = the 4 percentiles × channels {1, representative} × batch {8, 32, 64}, so 24 shapes, with duplicates removed); `FITS_SHAPE(meta) -> ProbeShape` (p90 length, 1 channel, batch 8); `run_probe(adapter_name, mode, shape, n_series=512, warmup_epochs=1, timed_epochs=3, grad_ckpt=False, device: str | None = None) -> RunRecord` (device None = acquire a GPU via `acquire_gpu`) (status `ok|oom|diverged|error`; `extra` has `s_per_epoch` (mean of the timed epochs), `s_per_1000_series`, `params_total`, `params_trainable`, `grad_ckpt`, and `inference_s_per_1000` from one no-grad pass over 1,000 synthetic series); CLI `python -m tor.probes.memtime --model NAME [--modes ...] [--fits-only]`.
  - When the run at `FITS_SHAPE` without checkpointing is `oom`, the runner retries that shape once with `grad_ckpt=True` and records both runs.
  - Synthetic data comes from `torch.randn` with seed 0 and random labels over 5 classes. No real data is involved.
- [ ] **Step 1: Write failing tests**:
  - `test_grid_size_and_fits_shape`: on the Task 5 fixture with representative channels 6, the grid has ≤ 24 unique shapes and `FITS_SHAPE == ProbeShape(p90, 1, 8)`.
  - `test_oom_recorded_and_grid_continues` (Review Focus 4): a fake adapter raises `torch.cuda.OutOfMemoryError` on batch 64 only. The run over 3 shapes yields statuses `[ok, ok, oom]` and `empty_cache` is called.
  - `test_ckpt_retry_on_fits_oom`: a fake that OOMs only without checkpointing produces 2 records at `FITS_SHAPE`, one `oom` with `grad_ckpt=False` and one `ok` with `grad_ckpt=True`.
  - `test_probe_uses_no_real_data`: monkeypatch `load_train` and `load_test` to raise, and `run_probe` on the toy adapter still succeeds (CPU allowed for tests via `device="cpu"`).
- [ ] **Step 2: Run tests.** Expected: FAIL.
- [ ] **Step 3: Implement**.
- [ ] **Step 4: Run tests.** Expected: 4 passed.
- [ ] **Step 5: Run the probes**, model by model, via `nohup` with at most 2 concurrent processes (each acquires its own GPU): `python -m tor.probes.memtime --model <name>` for every available adapter. Check `runs/runlog.jsonl` for one record per (model, mode, shape) and confirm no `error` statuses are left unexplained. Re-run any `shared=true` record at `FITS_SHAPE` when an idle card is free, and keep both records.
- [ ] **Step 6: Commit** `"Add probe runner and Phase 0 probe records"` (code and run log).

