### Task 4: Cost logger

**Files:** Create `tor/cost.py`, `tests/test_cost.py`

**Interfaces:**
- Consumes: `query_gpus` (Task 3).
- Produces: `measure(gpu_index: int | None, hosted: bool = False) -> ContextManager[CostRecord]`. `CostRecord` fields: `wall_s: float`, `peak_gpu_mem_mib: float | None` (from `torch.cuda.max_memory_allocated` after `reset_peak_memory_stats`), `gpu_model: str | None`, `cpu_cores: int` (= `len(os.sched_getaffinity(0))`), `torch_threads: int`, `load_start: float`, `load_end: float`, `other_pids_start: list[int]`, `other_pids_end: list[int]`, `shared: bool`, `hosted: bool`; `.to_dict()`. `LOAD_THRESHOLD = 48.0`.

- [ ] **Step 1: Write failing tests**:
  - `test_shared_false_when_alone`: the mocked GPU has no other pids and loadavg is 10, so `shared is False`.
  - `test_shared_true_when_other_pid_appears_at_end` (Review Focus 1): the pids snapshot at end is `[999]` (not ours), so `shared is True`.
  - `test_shared_true_on_high_load`: loadavg is 50.0, so `shared is True`.
  - `test_hosted_flag_passthrough`: `measure(None, hosted=True)` gives `hosted is True` and `peak_gpu_mem_mib is None`.
  - `test_wall_time_measured`: with `time.sleep(0.05)` inside, `wall_s >= 0.05`.
- [ ] **Step 2: Run tests.** Expected: FAIL.
- [ ] **Step 3: Implement**. Count as "other" any pid that is not in our own process tree (`os.getpid()` and its children via `/proc`). Load comes from `os.getloadavg()[0]`. Call `torch.set_num_threads(16)` at import if fewer threads aren't already set.
- [ ] **Step 4: Run tests.** Expected: 5 passed.
- [ ] **Step 5: Commit** `"Add cost logger with co-tenancy and hosted flags"`.

