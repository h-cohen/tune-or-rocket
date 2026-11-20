### Task 3: GPU picker with the sharing cap

**Files:** Create `tor/gpu.py`, `tests/test_gpu.py`

**Interfaces:**
- Produces: `GpuState(index:int, mem_used_mib:int, util_pct:int, other_pids:list[int])`; `query_gpus() -> list[GpuState]` (parses `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits` and `--query-compute-apps=gpu_uuid,pid`); `is_idle(g: GpuState) -> bool` (`mem_used_mib < 1024 and util_pct == 0`); `acquire_gpu(timeout_s: float = 3600, poll_s: float = 30, max_ours: int = 2) -> GpuLease` (a context manager with `.index`; it sets `CUDA_VISIBLE_DEVICES` and holds `fcntl.flock` on `TOR_CACHE/locks/gpu{i}.lock`); `NoGpuAvailable(Exception)`.

- [ ] **Step 1: Write failing tests** (all with `query_gpus` monkeypatched, no real GPU needed):
  - `test_is_idle_thresholds`: (1023 MiB, 0%) → True; (1024, 0) → False; (10, 1) → False.
  - `test_acquire_picks_idle_unlocked_gpu`: GPUs 0 and 1 are busy and 4 is idle, so `lease.index == 4` and `os.environ["CUDA_VISIBLE_DEVICES"] == "4"`.
  - `test_cap_of_two_leases`: 4 idle GPUs. Two leases are held, so a third with `timeout_s=0.1, poll_s=0.05` raises `NoGpuAvailable`.
  - `test_lock_released_when_holder_process_dies` (Review Focus 2): a subprocess acquires GPU 4 and exits without cleanup. The parent can then acquire GPU 4.
- [ ] **Step 2: Run** `pytest tests/test_gpu.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement**. To count our own leases, try a non-blocking flock on every `gpu*.lock`; each one that fails is held by us. Choose among idle GPUs that aren't locked, lowest index first.
- [ ] **Step 4: Run tests.** Expected: 4 passed. Also run `python -c "from tor.gpu import query_gpus; print(query_gpus())"` on the node. Expected: 8 entries.
- [ ] **Step 5: Commit** `"Add GPU picker with idle check and 2-lease cap"`.

