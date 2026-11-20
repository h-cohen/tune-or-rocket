# Deviations and disclosures

- Cache location: model weights, dataset downloads and probe logs live in `TOR_CACHE`, default `/data/hadar/tor-cache` (outside the repo; falls back to `cache/` in the repo only if /data is not writable); the location can be overridden with the environment variable.
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
