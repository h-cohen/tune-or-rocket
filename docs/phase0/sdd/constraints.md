Global constraints (verbatim from plan):
- No test split is ever read in Phase 0. `tor.data.load_test` raises `PhaseGuardError` when `TOR_PHASE=0`, with no exemption. `TOR_PHASE` defaults to `"0"`.
- No accuracy, F1 or validation score is computed or reported. Training loss may be logged only to show that a code path trains.
- At most 2 GPUs and 16 CPU cores in use at once. A GPU counts as idle only if it has < 1024 MiB used and 0% utilisation at launch.
- One GPU per run. No DataParallel or sharding.
- Secrets live only in `~/.config/tune-or-rocket/secrets.env` (`HF_TOKEN`, `TABPFN_TOKEN`). They are never logged, printed, written into run records or committed.
- Caches (HF models, datasets, locks) go under `TOR_CACHE`, default `/data/hadar/tor-cache`; fallback repo `cache/` if /data not writable.
- Every number in the report comes from a run record. A missing measurement appears as `missing: <reason>`. Statuses are `ok`, `oom`, `diverged`, `error`, `not_run: <reason>`.
- A run with another process on its GPU, or with a node 1-minute load average > 48, is marked `shared=true`. A hosted TabPFN run is marked `hosted=true`.
- "Fits" = the mode runs at batch 8, the 90th-percentile length and 1 channel, on one 48 GB card. Gradient checkpointing is allowed but recorded.
- Commits are authored by "Hadar Cohen" <hal.nls@gmail.com>, with plain messages and no co-author or session trailers.
- Public checkpoints and data only. The user's personal details never go into requests to external services.
Controller rulings binding on tasks:
- FMAdapter: class attrs `supported_modes`, `supports_grad_ckpt`, `lora_targets`; `classifier(n_classes, mode, n_channels, seq_len, grad_ckpt=False, seed=0)`; `is_available()` classmethod; `get_adapter(name)`; registry skips only ImportError/AdapterUnavailable.
- Env: conda env at /data/hadar/envs/tor (python 3.11). Run python/pytest via `/data/hadar/envs/tor/bin/python -m pytest`.
- Env note: conda install requires CONDA_NO_PLUGINS=true and --solver classic (libmamba broken). torch 2.14.1+cu126 installed.
