# Task 11a report: smoke run (code + tests)

Implemented tor/probes/smoke.py: choose_smoke_dataset (seeded 20261120; chosen dataset = ShapeletSim, n_train 20, declared pilot-exposed), smoke_config, run_smoke (full train split via support_draw, labels sorted -> 0..K-1, recipe epochs overridden and batch_size=min(recipe, n_train), status ok / diverged / error "loss did not decrease", extra with all requested fields + cost), run_plan (skips unsupported modes as "not_run: mode unsupported", resumable by config_hash over ok/oom/diverged, --force), CLI (--model --modes --epochs --jobs<=2 --timeout --force; prints dataset).
Reuse: memtime.isolated_runner / _child_main / _not_run generalised with optional run_cfg and kind params (defaults unchanged, probe behaviour identical).
Tests (tests/test_smoke.py, 5): the two brief tests, ok+fields, diverged-not-ok, plan skip/resume/force. Synthetic data, CPU, InceptionTime. test_smoke_never_reads_test patches load_test to raise (the one-epoch run returns status error "loss did not decrease", as one epoch cannot decrease).
Results: smoke+probes+runlog+cost+train+splits = 57 passed (226 s). Full suite exceeded 550 s under GPU contention, not completed. Isolated child on CPU with real ShapeletSim train split: ok, losses 0.717 -> 0.559.
TDD note: implementation was written before the RED run (no RED evidence).
