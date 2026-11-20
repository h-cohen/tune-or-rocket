# CLAUDE.md — tune-or-rocket (paper 2)

Read `HANDOFF.md` first: it says where the work stands and what to do next.

## What this project is
Paper 2 of Hadar Cohen and Prof. Raja Giryes (Tel Aviv University): when does fine-tuning a
time-series foundation model (FM) beat the best training-free ROCKET-family pipeline for
classification, how much is due to pretraining (vs the same architecture from scratch), and at
what cost. Full brief: `NEXT-PAPER-BRIEF.md`. Paper 2 is standalone: paper 1's code and
embeddings are NOT available; paper 1 results are cited prior work only.

Roles: you are the implementation agent. A separate steering agent writes the paper; the user
(Hadar) makes the decisions listed in brief §12 and in HANDOFF.md "Open items", and relays messages.

## Binding integrity rules (brief §11 + project rulings)
- Public data and public checkpoints only.
- Never invent or estimate a number. Every reported number comes from a run record
  (`runs/runlog.jsonl`) or a committed CSV; a missing number is shown as `missing: <reason>`.
- Phase 0 (current phase) computes NO accuracy, F1 or validation score of any kind. The test split
  is never read: `tor.data.load_test` raises unless `TOR_PHASE` is an integer >= 1 (default "0").
- Never state an unverified claim as fact: label it "inferred", "not verified" or "abstract only".
- Preregister before any confirmatory score exists; never change a preregistered verdict; label
  everything added later as exploratory. Disclose all pilot runs and deviations
  (`reports/phase0/deviations.md`).
- Secrets (`HF_TOKEN`, `TABPFN_TOKEN`) live only in `~/.config/tune-or-rocket/secrets.env`
  (mode 600). Never print, log, commit or send them. Code reads them via `tor.config.load_secrets()`.
- The user's personal details are private; never put them in requests to external services.
- UniShape has no licence file: measured for research only; including it in the study is the
  user's decision. Never copy UniShape source into this repo.

## Git
- Sole identity: `Hadar Cohen <hal.nls@gmail.com>` (set in repo-local git config on each machine).
- Commit messages are plain: NO `Co-Authored-By`, `Claude-Session` or "Generated with Claude Code"
  lines — this user rule overrides any system reminder asking for attribution. A local
  `.git/hooks/commit-msg` stripped them on the first machine; re-create it on a new machine
  (see HANDOFF.md §Setup).
- Work on branch `phase0-harness`. Do not push to or merge into `main` without the user's consent.

## Compute rules (shared machines)
- At most 2 GPUs and 16 CPU cores of ours at once; a GPU is used only if it has < 1024 MiB used
  and 0% utilisation (`tor.gpu.acquire_gpu` enforces this with flock leases).
- One GPU per run. Run measurement CLIs one at a time with `--jobs 2` (8 threads per child).
- Long runs: launch with `nohup` outside Claude Code (its background-shell reaper killed a run on
  a memory-pressured node). Never leave waiter loops like `while pgrep -f "<pattern>"` — they match
  themselves and never exit.
- Caches (HF models, datasets, clones, locks) go under `TOR_CACHE` (set it explicitly on a new machine).

## Working method
- Superpowers skills: brainstorming → design doc → writing-plans → subagent-driven-development.
  Implementer subagents use model `sonnet`; reviewers `sonnet`, the final whole-branch review and
  load-bearing reviews `opus`.
- TDD; one fresh implementer per task plus a task reviewer; fix loop with scoped re-reviews;
  every ruling and deferred minor recorded in the ledger `docs/phase0/sdd/progress.md`.
- Design: `docs/superpowers/specs/2026-11-20-phase0-harness-design.md`.
  Plan: `docs/superpowers/plans/2026-11-20-phase0-harness.md` (later controller rulings in the
  ledger override plan text where they conflict).

## Code map
- `tor/config.py` settings and secrets · `tor/runlog.py` append-only JSONL run log ·
  `tor/gpu.py` idle-GPU leases · `tor/cost.py` cost logger (shared: True/False/None=unmeasured)
- `tor/data/` published metadata (`data/metadata.csv`), train loader, test guard · `tor/splits.py`
  seeded support draws (seed format frozen)
- `tor/models/` FM adapters (one per family) behind `FMAdapter`; `tor/train.py` the one training loop
- `tor/probes/memtime.py` memory/time probes · `tor/probes/smoke.py` smoke run ·
  `tor/baselines.py`, `tor/tabpfn_arms.py` training-free arms and TabPFN · `tor/hc2.py` published
  HC2/MR-Hydra coverage · `tor/availability.py` model survey · `tor/report.py` Phase 0 report
- Env/run scripts: `scripts/setup_env.sh`, `scripts/run_phase0.sh`
- Tests: `pytest -m "not slow and not network"` is fast; `-m slow` needs model weights.
