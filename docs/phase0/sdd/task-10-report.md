# Task 10a report: probe runner (code + tests)

Commit: "Add probe runner". Step 5 (real GPU runs) not run.

Implemented
- tor/data/metadata.py: length_percentiles(meta, require_no_missing=True); False selects equal_length==True only.
- tor/probes/shapes.py: ProbeShape, probe_grid, FITS_SHAPE (both use require_no_missing=False).
- tor/probes/memtime.py: run_probe (in-process, device None = acquire_gpu; extra has all fields requested), isolated_runner (spawn child per cell; timeout clock starts once the child holds its GPU lease; timeout -> "not_run: timeout (N s)"; crash -> "error" with exit code), run_plan (FITS first for all model/mode, ckpt retry on oom or "not_run: no grad ckpt support", then rest of grid; unsupported modes recorded once; resumability via config_hash of ok/oom/diverged records; jobs capped at 2), CLI (--model NAME|all, --modes, --fits-only, --jobs, --timeout, --runlog). Entry points call limit_threads(16) and load_secrets().
- tests/test_probes.py (11 tests) + tests/probe_fakes.py (fake adapters; the sleeper registers itself in children via env TOR_PROBE_EXTRA_MODULES).
- pyproject: added tor.probes to packages.

Tests: 11 passed in tests/test_probes.py; 57 passed in the non-slow subset of the other test files. The full suite including slow contract tests was not completed (too slow here, stopped).
TDD note: I wrote implementation before tests this time; no RED run recorded.

Notes / concerns
- Timed-out and error cells are not skipped on resume (they are retried).
- Grid cells other than FITS are run without grad_ckpt only, per brief.
- A hung child killed by timeout may leave GPU memory until process exit (kill frees it).
- inference pass uses bf16 autocast on CUDA, batches of the probe batch size.

## Fix round 1
1. Children get threads = 16 // jobs (isolated_runner(threads=...), main computes it; jobs capped at 2); tests: CLI wiring (jobs 2 -> 8) and real child records torch_threads <= cap.
2. --rerun-shared (re-run cells whose existing records all have cost.shared != False; old kept) and --force; tests for True/None/False and force.
3. extra.peak_gpu_mem_reserved_mib (max_memory_reserved, same reset point as allocated; None on CPU).
4. Diverged runs drop the diverged epoch's time; s_per_epoch / s_per_1000_series are None when no complete timed epoch.
5. _is_oom also maps CUBLAS_STATUS_ALLOC_FAILED / CUDNN_STATUS_ALLOC_FAILED; get_device_name moved inside the try.
6. Child writes <out>.progress after each epoch (via new optional on_epoch callback in tor/train.py train_epochs); timeout record extra has epoch_times_s and partial (peaks).
7. OOM test asserts empty_cache follows the OOM event; new test for OOM in inference.
Tests: tests/test_probes.py 21 passed (foreground); tests/test_train.py passes.
