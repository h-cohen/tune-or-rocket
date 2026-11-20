# Task 14a report: Phase 0 report generator

Commit: "Add Phase 0 report generator" (tor/report.py, tests/test_report.py, reports/phase0/deviations.md). No rendered reports/phase0.md committed. runs/runlog.jsonl only read; no GPU use.

## Implemented
- model_table(records, availability, fits_shape, adapter_info): columns model, family, available, licence, params_total/unused/used, modes, fits_full/lora/head/scratch, ckpt_needed, s_per_epoch_fits (full), shared_flag (idle/shared/unmeasured), peak_alloc_mib and peak_reserved_mib (full), hosted, recipe_source, pretraining_corpora, notes. Availability CSV names mapped to adapter names via ADAPTER_FOR. Missing values rendered "missing: <reason>".
- pick(): completed record over not_run/error, idle over shared/unmeasured, latest otherwise.
- fits_* states: yes / yes (grad ckpt) / no (oom) / timeout (<s> s, partial epochs: ...) / not_run: <reason> / missing: no probe record.
- grid_completeness (planned vs ok/oom/diverged/timeout/error/not_run/no_record, per model; sums to planned), probe_grid_table (appendix CSV via --grid-csv), baseline_table, TabPFN limits renderer, render_report(), CLI `python -m tor.report [--out] [--grid-csv] [--runlog]`.
- Embedded: HC2 coverage/resample/usage sections (sanitised), literature "could not access" list, deviations.md, fixed Open items section (6 items).
- Literature is NOT inlined (it quotes MR-Hydra/HC2 scores); sanitize() masks score words in embedded text.

## Tests
tests/test_report.py: 13 tests (missing not imputed, unshared preferred, flags, fits states/ckpt, completed beats earlier timeout, params used, FITS-only timing, grid counts, no-score-words incl. adversarial embedded files, "missing: not yet run" sections, baselines/limits/deviations rendering, deviations entries, grid CSV). Full suite: 197 passed (8:36). TDD note: tests were written after the implementation in one pass (not strictly RED-first); one test failure along the way was a test-fixture issue (meta lacked a UEA row).

## Self-review / real-log render
Rendered against the current runlog to the scratchpad: 0 occurrences of accuracy/F1/val_. Current log: 29 probe records; FITS done for unishape, mantis8m/v2, nutime, moment-small; reverso full/lora timed out at 1800 s (the 5400 s rerun not in log yet).

## Concerns
- Deviations "TOR_CACHE location" worded neutrally (default /data/hadar/tor-cache; reason not in ledger).
- FITS-shape params can vary by mode (lora adds adapters); params are taken from full mode preferentially.
- params/s_per_epoch reflect the FITS shape only (length 1500 from current metadata).
- Open item wording for TIGER etc. is mine, condensed from literature section 4.

## Fix round 1
Fixed items 1-13: params semantics (used = probe params_total; checkpoint total = used + unused; source mode/run_id column; "0 (adapter reports none)"), deviations.md (TOR_CACHE deviation with reason, config_hash, require_no_missing 1500 vs 1496, MixedShapes alias), sanitize now excludes whole blocks (no rewriting), recipe_source in full, ckpt_needed missing unless all supported modes decisive, partial peaks as ">=n (partial, timeout)" with shared label always shown, chosen-record run_id column (+ chosen_for_cell in grid CSV), pick() latest for non-completed, ckpt retry status shown, smoke column renamed with derivation, n_series/epochs from record configs (classes = probe-runner constant, stated), docstring fix, default grid CSV path, baseline reason column.
Tests: tests/test_report.py 17 passed (foreground). Re-render against real log: 0 score words. Spot-check: moment-small total 35,339,973 (unused 16,449,536, used 18,890,437); chronos-bolt-small total 47,720,581 (unused 27,702,144, used 20,018,437; ckpt 47,718,016 plus head params); reverso (head-mode source) total 2,611,382 = used 2,446,853 + unused 164,529 (ckpt 2,609,585 plus head params; head record chosen because full timed out).
Note: reports/phase0/probe_grid.csv is generated, not committed.

## Fix round 2
Relabelled model-table param columns: "params_used (classifier incl. head)", "unused (removed from checkpoint)", "params_total (used + unused = checkpoint + head)". Section-2 derivation now says the total includes the freshly initialised head (size depends on n_classes=5, channels=1 of the probe) and that nothing unrecorded is subtracted. Removed unused `tag` variable. Test updated. tests/test_report.py: 17 passed (foreground). No render into the repo.
