# HANDOFF — Phase 0 of paper 2 (state at 2026-11-20, end of first session)

Read with `CLAUDE.md` (rules) and `NEXT-PAPER-BRIEF.md` (the study). Branch: `phase0-harness`.

## 1. Where we are

Phase 0 (brief §9) = checks only, no scores: literature, model availability, memory/time of
fine-tuning per model, TabPFN limits, HC2 availability → a report for the steering agent.

**All code of the plan is implemented, tested and reviewed** (Tasks 1–14 of
`docs/superpowers/plans/2026-11-20-phase0-harness.md`; each task had an implementer, a task review
and fix rounds; see the ledger `docs/phase0/sdd/progress.md` for every ruling and deferred minor).

| Plan task | State |
|---|---|
| 1–7 harness core (config, run log, GPU leases, cost, data + test guard, support draws, adapter interface + train loop + InceptionTime) | done |
| 8 model availability survey (`reports/phase0/models_availability.csv`, 15 rows, all reachable) | done |
| 9 FM adapters: mantis8m, mantisv2, moment-small/base/large, chronos-bolt-small/base, chronos2, timesfm25, toto, reverso, reverso-small, unishape, nutime (+ inceptiontime) | done |
| 10 probe runner (code) | done |
| 10 probe RUNS | **partial** (see §2) |
| 11 smoke run (code) | done — runs **not yet run** |
| 12 training-free arms + TabPFN (code) | done — timing runs **not yet run** |
| 13 HC2/MR-Hydra coverage (`reports/phase0/hc2_availability.md`) + literature (`reports/phase0/literature.md`) | done |
| 14 report generator (`tor/report.py`, `reports/phase0/deviations.md`) | code done — **final render not yet done** |
| Final whole-branch review + finish branch | **not done** |

## 2. Measurements so far (`runs/runlog.jsonl`, committed; hpc-node9, RTX A6000 48 GB, shared node)
- FITS shape (batch 8, length p90 = 1500, 1 channel, 512 synthetic series, 1 warmup + 3 timed
  epochs): every supported mode is `ok` for all 13 non-Reverso adapters (no OOM).
- Reverso: head, lora, full `ok` (lora/full measured with a 5400 s timeout; ~605–624 s/epoch,
  ≤ 3.6 GB reserved) but on a shared GPU → re-measure; scratch not yet measured.
- reverso-small: nothing measured yet.
- Smoke runs, CPU baseline timing, local/hosted TabPFN timing, TabPFN limits record: not run.
- The run driver was killed by Claude Code's memory-pressure reaper (other users' jobs used
  ~180 GB RAM), not by a bug.

## 3. What to do next (in order)
1. **Set up the machine** (§5). Run the fast tests: `pytest -q -m "not slow and not network"`.
   Then `pytest -q -m slow -k contract` once (downloads weights; CPU).
2. **Decide with the user about hardware**: the existing records are from hpc-node9 (A6000).
   If the new machine has different GPUs, either (a) re-run all probes there with `--force` so the
   timing table is single-hardware, or (b) keep both and report per `gpu_model`. Ask; do not mix silently.
   Also re-check the sharing cap (2 GPUs / 16 cores) for the new machine with the user.
3. **Run the remaining measurements**: `scripts/run_phase0.sh` (resumable; launch with `nohup`,
   outside Claude Code's background reaper). It does: Reverso + reverso-small FITS (timeout 5400 s)
   → smoke runs → CPU baselines + TabPFN limits → hosted TabPFN → re-run shared FITS cells → rest of
   the grid per model. The full grid (~1,400 cells) may not finish; that is allowed — the report
   shows completeness counts. Commit `runs/runlog.jsonl` after each step.
4. **Render the report** (plan Task 14 step 4): `python -m tor.report --out reports/phase0.md`
   (also writes `reports/phase0/probe_grid.csv`). Read it end to end; check every number traces to a
   record; commit both.
5. **Final whole-branch review** on the most capable model (superpowers
   subagent-driven-development `references/final-review.md`), pointing it at the deferred minors in
   the ledger; one fix wave; then superpowers:finishing-a-development-branch (ask the user before
   merging to main or opening a PR).
6. **Hand the report to the user** for the steering agent, with the open items below.

## 4. Open items for the user (decisions, not for the agent to settle)
- UniShape: no licence file in its repo (all rights reserved by default) — include it in the study?
- "Best training-free pipeline": TIGER (arXiv 2610.06156, reported above HC2 on 142 UCR) and other
  new training-free methods (see `reports/phase0/literature.md` §4) — add as arms?
- TabPFN version for the ROCKET+TabPFN arm: local v2.5 (RocketPFN) vs v3 vs hosted v3.5; hosted v3
  caps at 2000 columns, which breaks MASHT's 10000-feature budget on small datasets.
- Primary training-free baseline and equivalence margin (brief §12.4), hypotheses H1–H5 (§12.5).
- RocketPFN arXiv ID: brief says 2606.21786, notes say 2606.21776 — unresolved.
- Mantis v1 pretraining corpus: 1.89M (v2 text) vs 7M (v1 Sec. 2.4) — unreconciled. Mantis-8M,
  MOMENT, UniShape and NuTime saw UCR/UEA TRAIN splits in pretraining (exposure; brief §10).
- UEA HC2 published-results resample convention not verified (UCR's is, from Bake off redux).
- Smoke dataset ShapeletSim (seeded rule) is declared pilot-exposed.

## 5. Setup on a new machine
```bash
git clone https://github.com/h-cohen/tune-or-rocket.git && cd tune-or-rocket && git checkout phase0-harness
git config user.name "Hadar Cohen"; git config user.email "hal.nls@gmail.com"
# commit-msg hook that strips attribution trailers (user rule):
cat > .git/hooks/commit-msg <<'EOF'
#!/bin/sh
sed -i -e '/^Co-Authored-By:/Id' -e '/^Claude-Session:/d' -e '/Generated with \[Claude Code\]/d' "$1"
EOF
chmod +x .git/hooks/commit-msg
ENV_PREFIX=/path/envs/tor TOR_CACHE=/path/tor-cache bash scripts/setup_env.sh   # not yet tested end to end
# secrets: the user creates ~/.config/tune-or-rocket/secrets.env (chmod 600): HF_TOKEN=..., TABPFN_TOKEN=...
export TOR_CACHE=/path/tor-cache   # always set it; the default /data/hadar/tor-cache is hpc-node9-specific
```
Notes: several model packages are installed with `--no-deps` on purpose (their pins would downgrade
torch/transformers); `pip check` complaints about them are expected. `tor/config.py` hard-codes
`/data/hadar` as the default cache root with a repo `cache/` fallback — set `TOR_CACHE`.

## 6. Where things are
- Design / plan: `docs/superpowers/specs/…-design.md`, `docs/superpowers/plans/…-harness.md`
- Ledger of rulings, deviations, deferred minors, per-task status: `docs/phase0/sdd/progress.md`
- Per-task briefs and implementer reports (with fix rounds): `docs/phase0/sdd/task-*-brief.md`, `task-*-report.md`
- Shared implementer/reviewer instructions used: `docs/phase0/sdd/implementer-common.md`,
  `reviewer-common.md`, `task-9-common.md`, `task-9-lessons.md`, `constraints.md`
  (paths inside them point to the first machine and the gitignored `.superpowers/` workspace;
  re-create a workspace with the superpowers `sdd-workspace` script if continuing SDD)
- Literature notes (raw): `docs/phase0/sdd/lit-a..d.md`; merged: `reports/phase0/literature.md`
- Phase 0 outputs: `reports/phase0/` (availability CSV, HC2 coverage, literature, deviations)
