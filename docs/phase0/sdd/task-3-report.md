# Task 3 report
Implemented tor/gpu.py (GpuState, query_gpus, is_idle, acquire_gpu, GpuLease, NoGpuAvailable) and tests/test_gpu.py (5 tests: 4 specified + CUDA_VISIBLE_DEVICES restore).
Design: flock per gpu{i}.lock via separate open files; cap counted by non-blocking probes of gpu*.lock; count+acquire serialised by a blocking flock on locks/pick.mutex to avoid cap races between our processes.
TDD: RED = pytest tests/test_gpu.py -> ImportError (no tor.gpu). GREEN = full suite 17 passed.
Real node: query_gpus() returned 8 entries (read-only; no GPU held).
Concerns: probes briefly hold locks, so a concurrent probe may transiently over-count (conservative).
