# Task 4 report
Implemented tor/cost.py (CostRecord, measure, limit_threads, LOAD_THRESHOLD=48.0) and tests/test_cost.py (5 tests).
RED: pytest tests/test_cost.py -> ImportError: cannot import name 'cost' from 'tor'.
GREEN: full suite 22 passed (5 new).
Notes: no import-time set_num_threads; limit_threads(n=16) provided, torch_threads recorded at end. Own tree via /proc/*/stat ppid walk, computed at both snapshots. Torch device 0 used when CUDA_VISIBLE_DEVICES is a single index (documented in _torch_device). gpu_index None -> no nvidia-smi, GPU fields None. gpu_model from torch.get_device_name (GpuState has no name). Failure inside block still finalizes record (try/finally).

## Fix round 1
Changes in tor/cost.py: every measurement goes through `_try` (failure -> None/empty + reason in new `CostRecord.measure_errors`), so finalization never masks the block's exception. `shared` is `bool | None`: True on any evidence, None if co-tenancy unmeasured (unknown gpu index, query_gpus failure, loadavg failure) and no evidence, else False. `torch.cuda.is_available()` guard: if unavailable, peak/model None + reason. Added 4 tests (OOM not masked, unknown index -> None, CUDA unavailable, real child `sleep` pid not counted as other).
Command: /data/hadar/envs/tor/bin/python -m pytest tests/test_cost.py -q -> 9 passed; full suite 26 passed.
