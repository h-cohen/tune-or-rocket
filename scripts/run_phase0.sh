#!/bin/bash
# Phase 0 run driver. Runs the remaining measurement steps ONE CLI AT A TIME
# (thread cap: 16 cores total; --jobs 2 gives each child 8 threads; at most 2 GPUs).
# Every step is resumable: cells with ok/oom/diverged records are skipped; timeouts/errors are retried.
# Usage: PY=/path/to/envs/tor/bin/python TOR_CACHE=/path/to/tor-cache LOGS=/path/to/logs bash scripts/run_phase0.sh
# Run it outside Claude Code's background-shell reaper, e.g.:
#   nohup bash scripts/run_phase0.sh > "$LOGS/driver.out" 2>&1 &
cd "$(dirname "$0")/.."
PY="${PY:?set PY to the environment python}"
export TOR_CACHE="${TOR_CACHE:?set TOR_CACHE}"
L="${LOGS:-$TOR_CACHE/logs}"; mkdir -p "$L"
step(){ echo "== $* $(date -u +%FT%TZ)"; "$@"; echo "exit $? $(date -u +%FT%TZ)"; }

# 1. Remaining FITS cells. Reverso models use a 5400 s timeout (ruling: a LoRA epoch takes ~650 s).
step $PY -m tor.probes.memtime --model reverso --fits-only --jobs 2 --timeout 5400 >> $L/d-reverso.log 2>&1
step $PY -m tor.probes.memtime --model reverso-small --fits-only --jobs 2 --timeout 5400 >> $L/d-reverso-small.log 2>&1
# 2. Smoke runs (ShapeletSim training split, declared pilot-exposed).
step $PY -m tor.probes.smoke --model all --jobs 2 >> $L/d-smoke.log 2>&1
# 3. Training-free timing (CPU; nothing else may run concurrently) and TabPFN limits.
step $PY -m tor.baselines --which all --shapes fits,p50 --threads 16 >> $L/d-baselines.log 2>&1
# 4. Hosted TabPFN (user-approved; synthetic data only in Phase 0).
step $PY -m tor.baselines --which rocketpfn --shapes fits,p50 --hosted --threads 16 >> $L/d-hosted.log 2>&1
step $PY -m tor.baselines --which masht --shapes fits,p50 --hosted --threads 16 >> $L/d-hosted.log 2>&1
# 5. Re-measure FITS cells whose records were taken on a shared/unmeasured GPU.
step $PY -m tor.probes.memtime --model all --jobs 2 --rerun-shared --fits-only >> $L/d-rerun-shared.log 2>&1
# 6. Rest of the probe grid, non-Reverso models (Reverso grid only if time allows, with --timeout 5400).
for m in mantis8m mantisv2 nutime unishape inceptiontime moment-small moment-base chronos-bolt-small chronos-bolt-base chronos2 timesfm25 toto moment-large; do
  step $PY -m tor.probes.memtime --model $m --jobs 2 >> $L/d-grid-$m.log 2>&1
done
