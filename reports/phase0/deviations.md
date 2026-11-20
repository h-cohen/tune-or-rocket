# Deviations and disclosures

- Cache location (deviation from the design's repo-local `cache/`): model weights, dataset downloads and probe logs live in `TOR_CACHE`, default `/data/hadar/tor-cache`, because `/` had 63 GB free against 8 TB on `/data`; the repo `cache/` is only a fallback if `/data` is not writable, and the variable overrides both.
- Reverso per-run timeout is 5400 s instead of the default 1800 s: the LoRA FITS cell needed about 650 s per epoch and timed out at 1800 s; each timeout is written into the record status.
- Reverso timings reflect the pure-PyTorch reference implementation (Python-loop DeltaNet), not FlashFFTConv/fla kernels, and inputs are padded to 2048, so its compute is independent of series length.
- The harness trainer applies warmup only (no cosine or other decay schedule), while several model recipes (e.g. Reverso, NuTime) cite cosine schedules; clipping and per-parameter-group learning rates are not represented either.
- GPU memory peaks exclude the CUDA context (about 0.5 GB). On real data the peak exceeds the probe peak by (N-512)*C*L*4 bytes, because the trainer moves all of X to the device.
- Inference timing has no warmup pass, so the first-call overhead is included in `inference_s_per_1000`.
- Probes use synthetic data only (random series, 5 classes, 512 training series, 1 warmup + 3 timed epochs); times are per-epoch means over the timed epochs.
- MASHT uses n_estimators 8 instead of the paper's 88 (the extracted appendix lists 88, possibly a typo; unverified).
- The smoke dataset (ShapeletSim, chosen by a seeded draw, training split only) is declared pilot-exposed: treat it as seen in any later selection or reporting.
- The probe grid may be partial (about 1,400 cells would take days on two shared GPUs); the grid-completeness table lists exactly which cells ran, and fits-shape cells ran first.
- Literature sources were not all accessible (OpenReview TS-LIMITS, ACML FMLTS list, Semantic Scholar, Google Scholar); absence of hits is not proof of absence.
- `config_hash` excludes the git commit and the device, so resumed runs keep records made at earlier commits; `--force` re-runs a cell. Each record still stores its own commit.
- FITS and probe lengths use `require_no_missing=False` (equal-length datasets including unknown missingness; missingness does not change compute shape): p90 is 1500, versus 1496 with the no-missing filter.
- Metadata alias: `MixedShapesRegularTrain` (archive name) maps to `MixedShapes` in the published metadata CSV (sizes and length match).
- Thread cap exceeded in the first FITS pass: two probe CLIs ran side by side with `--jobs 1` each (16 intra-op threads each), so FITS records written before 2026-11-20 ~14:00 UTC on hpc-node9 may have run with up to 32 threads on a 16-core budget; their CPU-side timing may be inflated. Later runs use one CLI at a time with `--jobs 2` (8 threads per child). Re-measure with `--force --fits-only` if this matters.
- Hardware: Phase 0 records so far come from hpc-node9 (RTX A6000 48 GB, shared node). Records from a different machine carry their own `gpu_model`; do not mix hardware in one timing table without saying so.
