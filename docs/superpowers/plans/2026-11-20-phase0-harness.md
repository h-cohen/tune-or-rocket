# Phase 0 and Core Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `tor` harness (data on official splits, support draws, FM adapters, one training loop, training-free arms, cost logging, GPU sharing, run log) and use it to produce the Phase 0 report, with no accuracy scores of any kind.

**Architecture:** A Python package `tor/` with one module per responsibility. Every measurement goes through `tor.cost.measure` and lands as one JSON line in `runs/runlog.jsonl`. Each FM is wrapped by an adapter class behind one interface, so the probe runner and the smoke run don't depend on any particular model. The report is generated from the run log plus two hand-written sections (literature, HC2 availability).

**Tech Stack:** Python 3.11 (conda env at `/data/hadar/envs/tor`), PyTorch (CUDA build matching driver 580 / CUDA 13), `peft`, `aeon`, `tabpfn` (local) and `tabpfn-client` (hosted), `huggingface_hub`, `pytest`. Model packages are added by the task that needs them: `mantis-tsfm`, `momentfm`, `chronos-forecasting`, and others after verification.

**Spec:** `docs/superpowers/specs/2026-11-20-phase0-harness-design.md` (and `NEXT-PAPER-BRIEF.md` for context).

## Global Constraints

- No test split is ever read in Phase 0. `tor.data.load_test` raises `PhaseGuardError` when `TOR_PHASE=0`, with no exemption. `TOR_PHASE` defaults to `"0"`.
- No accuracy, F1 or validation score is computed or reported. Training loss may be logged only to show that a code path trains.
- At most 2 GPUs and 16 CPU cores in use at once. A GPU counts as idle only if it has < 1024 MiB used and 0% utilisation at launch.
- One GPU per run. No DataParallel or sharding.
- Secrets live only in `~/.config/tune-or-rocket/secrets.env` (`HF_TOKEN`, `TABPFN_TOKEN`). They are never logged, printed, written into run records or committed.
- Caches (HF models, datasets, locks) go under `TOR_CACHE`, default `/data/hadar/tor-cache`. *(Plan call: `/data` has 8 TB free; `/` has 63 GB. The spec's repo-local `cache/` remains the fallback when `/data` isn't writable.)*
- Every number in the report comes from a run record. A missing measurement appears as `missing: <reason>`. Statuses are `ok`, `oom`, `diverged`, `error`, `not_run: <reason>`.
- A run with another process on its GPU, or with a node 1-minute load average > 48, is marked `shared=true`. A hosted TabPFN run is marked `hosted=true`.
- "Fits" = the mode runs at batch 8, the 90th-percentile length and 1 channel, on one 48 GB card. Gradient checkpointing is allowed but recorded.
- Commits are authored by "Hadar Cohen" <hal.nls@gmail.com>, with plain messages and no co-author or session trailers.
- Public checkpoints and data only. The user's personal details never go into requests to external services.

## Review Focus

1. **A GPU that becomes busy mid-run** (another user starts a job): the run must still finish, and the record must come out `shared=true` because co-tenancy is sampled at both start and end. Task 4 tests the end-snapshot path.
2. **A crashed or killed run holding a GPU lock**: the slot must free itself (flock is released when the process dies), not block forever. Task 3 tests a lock taken in a subprocess that then exits.
3. **A dataset with fewer than n training series in some class** (n = 5 or 20): `support_draw` must raise `InsufficientClassError` naming the class. It must not silently take fewer. Task 6 tests this.
4. **A model that OOMs at one probe shape**: the runner must record `oom` for that cell, free memory (`torch.cuda.empty_cache`) and continue with the grid instead of dying. Task 9 tests this with a fake adapter that raises `torch.cuda.OutOfMemoryError`.
5. **A gated checkpoint whose licence is not accepted** (HF 401/403): the availability survey records `not_run: gated licence`, the adapter registry skips that model, and the probe runner doesn't crash. Task 8 tests this with a mocked `huggingface_hub` error.

---

### Task 1: Project scaffold, environment, config and secrets

**Files:**
- Create: `pyproject.toml`, `tor/__init__.py`, `tor/config.py`, `tests/test_config.py`, `requirements.lock`, `README.md` (append a "Setup" section)

**Interfaces:**
- Produces: `tor.config.Settings` (frozen dataclass: `phase: str`, `cache_dir: Path`, `repo_root: Path`, `runlog_path: Path` = `repo_root/"runs/runlog.jsonl"`); `get_settings() -> Settings`; `load_secrets(path: Path = ~/.config/tune-or-rocket/secrets.env) -> None`, which sets `HF_TOKEN` and `TABPFN_TOKEN` in `os.environ` only if they are unset; `redact(s: str) -> str`, which replaces any loaded secret value with `***`.

- [ ] **Step 1: Create env** — `conda create -y -p /data/hadar/envs/tor python=3.11`. Then install a torch CUDA wheel and `peft aeon huggingface_hub pytest`. Run `pip install -e .`. Write `pip freeze` to `requirements.lock`.
- [ ] **Step 2: Write failing tests** in `tests/test_config.py`:
  - `test_phase_defaults_to_zero`: with `TOR_PHASE` unset, `get_settings().phase == "0"`.
  - `test_load_secrets_sets_env_without_overwriting`: tmp env file `HF_TOKEN=abc\nTABPFN_TOKEN=xyz`. After the load, `os.environ["HF_TOKEN"] == "abc"`. A preset `TABPFN_TOKEN=keep` stays `"keep"`.
  - `test_redact_masks_secret_values`: after the load, `redact("token abc here") == "token *** here"`.
  - `test_cache_dir_falls_back_when_data_unwritable`: monkeypatch `os.access` to False for `/data/hadar`, so `cache_dir == repo_root/"cache"`.
- [ ] **Step 3: Run** `pytest tests/test_config.py -v`. Expected: FAIL (module missing).
- [ ] **Step 4: Implement** `tor/config.py`. Read `TOR_CACHE` and `TOR_PHASE` from the environment.
- [ ] **Step 5: Run** `pytest tests/test_config.py -v`. Expected: 4 passed.
- [ ] **Step 6: Commit** `git add pyproject.toml tor tests requirements.lock README.md && git commit -m "Add package scaffold, settings and secret loading"`

### Task 2: Run log

**Files:** Create `tor/runlog.py`, `tests/test_runlog.py`

**Interfaces:**
- Consumes: `get_settings`, `redact` (Task 1).
- Produces: `config_hash(config: dict) -> str` (sha256 of canonical JSON with sorted keys, first 16 hex characters); `RunRecord` dataclass with fields `run_id: str`, `kind: str` (`probe|smoke|baseline|tabpfn_limits|availability`), `config: dict`, `config_hash: str`, `status: str`, `cost: dict`, `extra: dict`, `git_commit: str`, `versions: dict[str,str]`, `started: str` (ISO UTC); `append(record: RunRecord, path: Path | None = None) -> None`; `read_all(path: Path | None = None) -> list[RunRecord]`.

- [ ] **Step 1: Write failing tests**:
  - `test_config_hash_is_order_independent`: `config_hash({"a":1,"b":2}) == config_hash({"b":2,"a":1})`, and the length is 16.
  - `test_append_then_read_roundtrip`: append 2 records to tmp_path, then `read_all` returns 2 with equal fields.
  - `test_append_redacts_secrets`: with `HF_TOKEN=abc` loaded and a record whose `extra={"msg":"abc"}`, the line written to the file doesn't contain `"abc"`.
  - `test_status_must_be_valid`: `RunRecord(status="great", ...)` raises `ValueError`. Valid values are `ok`, `oom`, `diverged`, `error`, and anything starting with `not_run: `.
- [ ] **Step 2: Run** `pytest tests/test_runlog.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement**. `git_commit` comes from `git rev-parse HEAD` and gets suffix `+dirty` if `git status --porcelain` is non-empty. `versions` covers torch, aeon, peft, tabpfn, plus the model package if one is imported. Each write is a single `write()` of one line with `\n`, in append mode, so concurrent runs don't interleave.
- [ ] **Step 4: Run tests.** Expected: 4 passed.
- [ ] **Step 5: Commit** `"Add append-only run log"`.

### Task 3: GPU picker with the sharing cap

**Files:** Create `tor/gpu.py`, `tests/test_gpu.py`

**Interfaces:**
- Produces: `GpuState(index:int, mem_used_mib:int, util_pct:int, other_pids:list[int])`; `query_gpus() -> list[GpuState]` (parses `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits` and `--query-compute-apps=gpu_uuid,pid`); `is_idle(g: GpuState) -> bool` (`mem_used_mib < 1024 and util_pct == 0`); `acquire_gpu(timeout_s: float = 3600, poll_s: float = 30, max_ours: int = 2) -> GpuLease` (a context manager with `.index`; it sets `CUDA_VISIBLE_DEVICES` and holds `fcntl.flock` on `TOR_CACHE/locks/gpu{i}.lock`); `NoGpuAvailable(Exception)`.

- [ ] **Step 1: Write failing tests** (all with `query_gpus` monkeypatched, no real GPU needed):
  - `test_is_idle_thresholds`: (1023 MiB, 0%) → True; (1024, 0) → False; (10, 1) → False.
  - `test_acquire_picks_idle_unlocked_gpu`: GPUs 0 and 1 are busy and 4 is idle, so `lease.index == 4` and `os.environ["CUDA_VISIBLE_DEVICES"] == "4"`.
  - `test_cap_of_two_leases`: 4 idle GPUs. Two leases are held, so a third with `timeout_s=0.1, poll_s=0.05` raises `NoGpuAvailable`.
  - `test_lock_released_when_holder_process_dies` (Review Focus 2): a subprocess acquires GPU 4 and exits without cleanup. The parent can then acquire GPU 4.
- [ ] **Step 2: Run** `pytest tests/test_gpu.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement**. To count our own leases, try a non-blocking flock on every `gpu*.lock`; each one that fails is held by us. Choose among idle GPUs that aren't locked, lowest index first.
- [ ] **Step 4: Run tests.** Expected: 4 passed. Also run `python -c "from tor.gpu import query_gpus; print(query_gpus())"` on the node. Expected: 8 entries.
- [ ] **Step 5: Commit** `"Add GPU picker with idle check and 2-lease cap"`.

### Task 4: Cost logger

**Files:** Create `tor/cost.py`, `tests/test_cost.py`

**Interfaces:**
- Consumes: `query_gpus` (Task 3).
- Produces: `measure(gpu_index: int | None, hosted: bool = False) -> ContextManager[CostRecord]`. `CostRecord` fields: `wall_s: float`, `peak_gpu_mem_mib: float | None` (from `torch.cuda.max_memory_allocated` after `reset_peak_memory_stats`), `gpu_model: str | None`, `cpu_cores: int` (= `len(os.sched_getaffinity(0))`), `torch_threads: int`, `load_start: float`, `load_end: float`, `other_pids_start: list[int]`, `other_pids_end: list[int]`, `shared: bool`, `hosted: bool`; `.to_dict()`. `LOAD_THRESHOLD = 48.0`.

- [ ] **Step 1: Write failing tests**:
  - `test_shared_false_when_alone`: the mocked GPU has no other pids and loadavg is 10, so `shared is False`.
  - `test_shared_true_when_other_pid_appears_at_end` (Review Focus 1): the pids snapshot at end is `[999]` (not ours), so `shared is True`.
  - `test_shared_true_on_high_load`: loadavg is 50.0, so `shared is True`.
  - `test_hosted_flag_passthrough`: `measure(None, hosted=True)` gives `hosted is True` and `peak_gpu_mem_mib is None`.
  - `test_wall_time_measured`: with `time.sleep(0.05)` inside, `wall_s >= 0.05`.
- [ ] **Step 2: Run tests.** Expected: FAIL.
- [ ] **Step 3: Implement**. Count as "other" any pid that is not in our own process tree (`os.getpid()` and its children via `/proc`). Load comes from `os.getloadavg()[0]`. Call `torch.set_num_threads(16)` at import if fewer threads aren't already set.
- [ ] **Step 4: Run tests.** Expected: 5 passed.
- [ ] **Step 5: Commit** `"Add cost logger with co-tenancy and hosted flags"`.

### Task 5: Data: published metadata, train loading, test guard

**Files:** Create `tor/data/__init__.py`, `tor/data/metadata.py`, `tor/data/load.py`, `tests/test_data.py`, `data/metadata.csv` (generated and committed)

**Interfaces:**
- Produces: `build_metadata() -> pandas.DataFrame` with columns `name, archive (UCR|UEA), n_train, n_test, length (int or NaN if unequal), channels, classes, equal_length: bool, missing: bool, source_url`; `load_metadata() -> DataFrame` (reads `data/metadata.csv`); `length_percentiles(meta) -> dict` with keys `p10, p50, p90, max` (ints, over rows where `equal_length & ~missing`, UCR and UEA together); `representative_channels(meta) -> int` (median `channels` over UEA rows with `channels > 1`, rounded); `load_train(name: str) -> tuple[np.ndarray, np.ndarray]` (X shape `(n, c, L)`, y as str labels); `load_test(name)`; `PhaseGuardError(Exception)`.

- [ ] **Step 1: Write failing tests**:
  - `test_load_test_blocked_in_phase0`: with `TOR_PHASE=0`, `load_test("GunPoint")` raises `PhaseGuardError`, and the message contains `"Phase 0"`.
  - `test_load_test_guard_runs_before_any_io`: monkeypatch `aeon.datasets.load_classification` to raise `AssertionError`. With `TOR_PHASE=0` the call still raises `PhaseGuardError`, which proves the guard runs first.
  - `test_load_train_shape` (marker `network`): `X, y = load_train("GunPoint")` gives `X.shape == (50, 1, 150)` and `len(set(y)) == 2`.
  - `test_length_percentiles_on_fixture`: a 5-row fixture with lengths [10,20,30,40,1000] gives `max == 1000` and `p50 == 30`.
  - `test_metadata_has_no_test_derived_reads`: monkeypatch `load_test` to raise, then call `build_metadata()` (marker `network`). It succeeds.
- [ ] **Step 2: Run** `pytest tests/test_data.py -v -m "not network"`. Expected: FAIL.
- [ ] **Step 3: Implement**. `load_train` calls `aeon.datasets.load_classification(name, split="train", extract_path=TOR_CACHE/"datasets")`. `build_metadata` uses published summaries only. First try aeon's metadata helper (verify the name in the installed aeon; `aeon.datasets` has dataset lists and a metadata accessor in recent versions). Fall back to the timeseriesclassification.com dataset-summary CSV (verify the URL), and record which one was used in `source_url`. If neither gives `n_test` for a dataset, store NaN. Never compute it from files.
- [ ] **Step 4: Run** `pytest tests/test_data.py -v`. Expected: 5 passed. Then `python -m tor.data.metadata` writes `data/metadata.csv`. Check `wc -l data/metadata.csv`: about 128 UCR + 30 UEA rows (report the actual count).
- [ ] **Step 5: Commit** `"Add metadata from published summaries, train loader and test guard"`.

### Task 6: Support draws

**Files:** Create `tor/splits.py`, `tests/test_splits.py`

**Interfaces:**
- Consumes: `load_train` (Task 5).
- Produces: `Budget = tuple[Literal["per_class","fraction"], float]`. Constants: `BUDGETS = [("per_class",5),("per_class",20),("fraction",0.1),("fraction",1.0)]`, `DRAWS = range(10)`. Also `draw_seed(dataset: str, budget: Budget, draw: int) -> int` (first 8 bytes of sha256 of `f"{dataset}|{kind}|{value}|{draw}"`, as an int); `support_draw(y: np.ndarray, budget: Budget, draw: int, dataset: str) -> np.ndarray` (sorted train indices); `InsufficientClassError(ValueError)`; `save_draws(dataset: str, y) -> Path` (writes `splits/{dataset}.json`: `{budget_key: {draw: [indices]}}`).

- [ ] **Step 1: Write failing tests**:
  - `test_per_class_exact_counts`: y has 3 classes × 30. `("per_class",5)` gives 15 indices, with 5 from each class.
  - `test_deterministic`: two calls with the same args return identical arrays. Draws 0 and 1 differ.
  - `test_fraction_stratified_at_least_one_per_class`: y has class counts [100, 3]. `("fraction",0.1)` gives 10 + 1 indices, so each class has ≥ 1 (round each class count, with a minimum of 1).
  - `test_full_budget_is_all_indices`: `("fraction",1.0)` gives `arange(len(y))`.
  - `test_insufficient_class_raises` (Review Focus 3): class counts [30, 4] with `("per_class",5)` raise `InsufficientClassError` whose message contains the label of the class with 4.
- [ ] **Step 2: Run tests.** Expected: FAIL.
- [ ] **Step 3: Implement** with `numpy.random.default_rng(draw_seed(...))`, sampling within each class without replacement.
- [ ] **Step 4: Run tests.** Expected: 5 passed.
- [ ] **Step 5: Commit** `"Add seeded support draws from the training split"`.

### Task 7: Adapter interface, training loop, InceptionTime reference

**Files:** Create `tor/models/__init__.py` (registry), `tor/models/base.py`, `tor/models/inceptiontime.py`, `tor/train.py`, `tests/test_train.py`, `tests/models/test_contract.py`

**Interfaces:**
- Produces:
  - `Mode = Literal["head","lora","full","scratch"]`.
  - `class FMAdapter(ABC)`: class attrs `name: str`, `hf_repo: str | None`, `max_length: int | None`, `multivariate: Literal["native","per_channel"]`, `pretraining_corpora: str` (cited), `recipe_source: str`; methods `load(init: Literal["pretrained"] | tuple[Literal["random"], int]) -> None`, `preprocess(X: np.ndarray) -> torch.Tensor`, `embed(X: torch.Tensor) -> torch.Tensor` (shape `(n, d)`), `classifier(n_classes: int, mode: Mode, grad_ckpt: bool = False) -> torch.nn.Module` (the `scratch` mode loads `("random", seed)` first, then trains like `full`), `default_recipe() -> Recipe`, `lora_config() -> peft.LoraConfig` (default `r=8, lora_alpha=16, target_modules` = the model's attention q/k/v/o projection names, dropout 0.0).
  - `Recipe(lr: float, weight_decay: float, epochs: int, batch_size: int, warmup_frac: float = 0.0)`.
  - `REGISTRY: dict[str, type[FMAdapter]]`; `register(cls)` decorator; `available_adapters() -> list[str]` (skips any adapter whose import or availability check fails, logging the reason).
  - `count_params(module) -> tuple[int, int]` (total, trainable).
  - `train_epochs(model, X: torch.Tensor, y: torch.Tensor, recipe: Recipe, epochs: int, device: str, amp: bool = True) -> TrainResult(losses: list[float], epoch_times_s: list[float], diverged: bool)`, with AdamW, cross-entropy and bf16 autocast if `amp`. `diverged` is True on a non-finite loss, and training stops at that point. No evaluation function exists in this module.
  - `InceptionTime` adapter (`name="inceptiontime"`, in PyTorch: 6 inception modules of depth 3, 32 filters, kernel sizes 40/20/10, residual every 3, GAP head). Here `pretrained` is not supported and only the `scratch` mode exists.
- [ ] **Step 1: Write failing tests** in `tests/test_train.py`, using a tiny `ToyAdapter` defined in the test (a linear encoder):
  - `test_modes_set_trainable_params`: in `head` mode only the head params train; in `full` mode they all do; `count_params` reflects this.
  - `test_train_epochs_records_times_and_losses`: 3 epochs on random data give `len(losses) == 3`, `len(epoch_times_s) == 3` and `diverged is False`.
  - `test_divergence_detected`: lr=1e6 on toy data gives `diverged is True`, with fewer than the requested epochs recorded.
  - `test_no_eval_api`: `not hasattr(tor.train, "evaluate")` and `not hasattr(tor.train, "accuracy")`.
- [ ] **Step 2: Write** `tests/models/test_contract.py`, parametrized over `available_adapters()` and marked `slow`. For each adapter and each mode it supports: `classifier(3, mode)` on `preprocess(randn(4,1,128))` returns logits of shape `(4,3)`; the trainable fraction matches the mode (`head` < `lora` < `full`); `embed` returns 2-D. InceptionTime must pass now.
- [ ] **Step 3: Run** `pytest tests/test_train.py tests/models -v`. Expected: FAIL.
- [ ] **Step 4: Implement** the base class, registry, loop and InceptionTime.
- [ ] **Step 5: Run** `pytest tests/test_train.py tests/models -v -m "slow or not slow"`. Expected: all passed.
- [ ] **Step 6: Commit** `"Add adapter interface, training loop and InceptionTime reference"`.

### Task 8: Model availability survey

**Files:** Create `tor/availability.py`, `tests/test_availability.py`, `reports/phase0/models_availability.csv` (generated)

**Interfaces:**
- Consumes: `append` (Task 2), `load_secrets` (Task 1).
- Produces: `CANDIDATES: list[Candidate(name, family: Literal["classification","general","forecaster","deep_ref"], hf_repo: str | None, code_url: str | None, paper: str)]`, holding UniShape, Mantis-8M, MantisV2, NuTime, MOMENT-1-small/base/large, Chronos-Bolt-small/base, Chronos-2, TimesFM-2.5, Toto, Reverso and InceptionTime. Repo IDs are to be verified; a candidate with no repo found gets `hf_repo=None`. Also `check(c: Candidate) -> dict` with keys `available: bool`, `licence: str | None`, `gated: bool`, `status: str`, `files_gb: float | None`; and `survey() -> DataFrame` (writes the CSV and one `kind="availability"` run record per candidate).
- [ ] **Step 1: Write failing tests** (with `huggingface_hub` mocked):
  - `test_gated_without_acceptance_recorded` (Review Focus 5): `hf_hub_download` raises `GatedRepoError`, so `status == "not_run: gated licence"` and `available is False`.
  - `test_missing_repo_recorded`: `hf_repo=None` gives `status == "not_run: no public weights found"`.
  - `test_licence_read_from_card`: the mocked `model_info().card_data.license == "apache-2.0"` gives `licence == "apache-2.0"`.
- [ ] **Step 2: Run tests.** Expected: FAIL.
- [ ] **Step 3: Implement**. Use `huggingface_hub.model_info` for the licence, gating and file sizes, and download `config.json` only to prove access. For UniShape, NuTime and Reverso, search HF and GitHub by paper title. Record what was found and where, and leave a candidate as `None` rather than guessing a repo.
- [ ] **Step 4: Run tests**, then `python -m tor.availability`. Expected: a CSV with 14 rows, each with a status. Report the gated rows to the user so they can accept the licences.
- [ ] **Step 5: Commit** `"Add model availability survey"` (CSV included).

### Task 9: FM adapters (one sub-commit per family)

**Files:** Create `tor/models/mantis.py`, `tor/models/moment.py`, `tor/models/chronos.py` (Bolt + Chronos-2), `tor/models/timesfm.py`, `tor/models/toto.py`, `tor/models/reverso.py`, `tor/models/unishape.py`, `tor/models/nutime.py`. Write only the ones Task 8 marked available.

**Interfaces:**
- Consumes: `FMAdapter`, `register`, `Recipe` (Task 7).
- Produces: registered adapters named `mantis8m, mantisv2, moment-small, moment-base, moment-large, chronos-bolt-small, chronos-bolt-base, chronos2, timesfm25, toto, reverso, unishape, nutime`.

For each family, in order Mantis → MOMENT → Chronos → the rest:
- [ ] **Step A: Install** the model's package into the env, then refresh `requirements.lock`.
- [ ] **Step B: Implement the adapter.** `embed` uses the model's own classification or embedding path if it has one (Mantis: the model's output token; MOMENT: the input to its classification head, i.e. mean over patch embeddings). Otherwise it uses the encoder output mean-pooled over tokens (Chronos-Bolt: encoder only, as Jain et al. do). Multichannel data goes per-channel with embeddings concatenated, unless the model is natively multivariate (Chronos-2, Toto). Series longer than `max_length` are resized or truncated the way the model's own code does it; document this in the docstring with a link to the source. `default_recipe()` cites its source. Where none is published, use `Recipe(lr=1e-4, weight_decay=0.01, epochs=100, batch_size=32)` and set `recipe_source="harness default"`.
- [ ] **Step C: Run** `pytest tests/models/test_contract.py -k <name> -m slow -v` on an acquired GPU. Expected: pass for head, lora, full and scratch. A mode the model can't support gets `pytest.skip` with the reason, and that reason goes into the report.
- [ ] **Step D: Commit** `"Add <family> adapter"`.

### Task 10: Memory and time probe runner

**Files:** Create `tor/probes/__init__.py`, `tor/probes/shapes.py`, `tor/probes/memtime.py`, `tests/test_probes.py`

**Interfaces:**
- Consumes: `load_metadata`, `length_percentiles`, `representative_channels` (Task 5); `REGISTRY`, `train_epochs` (Task 7); `acquire_gpu` (Task 3); `measure` (Task 4); `append` (Task 2).
- Produces: `ProbeShape(length:int, channels:int, batch:int)`; `probe_grid(meta) -> list[ProbeShape]` (lengths = the 4 percentiles × channels {1, representative} × batch {8, 32, 64}, so 24 shapes, with duplicates removed); `FITS_SHAPE(meta) -> ProbeShape` (p90 length, 1 channel, batch 8); `run_probe(adapter_name, mode, shape, n_series=512, warmup_epochs=1, timed_epochs=3, grad_ckpt=False, device: str | None = None) -> RunRecord` (device None = acquire a GPU via `acquire_gpu`) (status `ok|oom|diverged|error`; `extra` has `s_per_epoch` (mean of the timed epochs), `s_per_1000_series`, `params_total`, `params_trainable`, `grad_ckpt`, and `inference_s_per_1000` from one no-grad pass over 1,000 synthetic series); CLI `python -m tor.probes.memtime --model NAME [--modes ...] [--fits-only]`.
  - When the run at `FITS_SHAPE` without checkpointing is `oom`, the runner retries that shape once with `grad_ckpt=True` and records both runs.
  - Synthetic data comes from `torch.randn` with seed 0 and random labels over 5 classes. No real data is involved.
- [ ] **Step 1: Write failing tests**:
  - `test_grid_size_and_fits_shape`: on the Task 5 fixture with representative channels 6, the grid has ≤ 24 unique shapes and `FITS_SHAPE == ProbeShape(p90, 1, 8)`.
  - `test_oom_recorded_and_grid_continues` (Review Focus 4): a fake adapter raises `torch.cuda.OutOfMemoryError` on batch 64 only. The run over 3 shapes yields statuses `[ok, ok, oom]` and `empty_cache` is called.
  - `test_ckpt_retry_on_fits_oom`: a fake that OOMs only without checkpointing produces 2 records at `FITS_SHAPE`, one `oom` with `grad_ckpt=False` and one `ok` with `grad_ckpt=True`.
  - `test_probe_uses_no_real_data`: monkeypatch `load_train` and `load_test` to raise, and `run_probe` on the toy adapter still succeeds (CPU allowed for tests via `device="cpu"`).
- [ ] **Step 2: Run tests.** Expected: FAIL.
- [ ] **Step 3: Implement**.
- [ ] **Step 4: Run tests.** Expected: 4 passed.
- [ ] **Step 5: Run the probes**, model by model, via `nohup` with at most 2 concurrent processes (each acquires its own GPU): `python -m tor.probes.memtime --model <name>` for every available adapter. Check `runs/runlog.jsonl` for one record per (model, mode, shape) and confirm no `error` statuses are left unexplained. Re-run any `shared=true` record at `FITS_SHAPE` when an idle card is free, and keep both records.
- [ ] **Step 6: Commit** `"Add probe runner and Phase 0 probe records"` (code and run log).

### Task 11: End-to-end smoke run on a real training split

**Files:** Create `tor/probes/smoke.py`, `tests/test_smoke.py`

**Interfaces:**
- Consumes: `load_metadata`, `load_train` (Task 5); `support_draw` (Task 6); registry and `train_epochs` (Task 7); `acquire_gpu`, `measure`, `append`.
- Produces: `choose_smoke_dataset(meta) -> str`, a seeded rule: candidates are UCR rows with `equal_length & ~missing & n_train <= 500 & length <= 512 & classes <= 10`, sorted by name, and `numpy.random.default_rng(20261120).choice` picks one. Also `run_smoke(adapter_name, mode, epochs=5, device: str | None = None) -> RunRecord` (`kind="smoke"`, full training split, budget `("fraction",1.0)`; `extra` has `losses` and `dataset`; status `ok` requires finite losses and a last loss below the first).
- [ ] **Step 1: Write failing tests**:
  - `test_smoke_choice_is_deterministic_and_eligible`: on the committed `data/metadata.csv`, two calls return the same name, and that row meets every filter.
  - `test_smoke_never_reads_test`: with `load_test` monkeypatched to raise, `run_smoke("inceptiontime","scratch",epochs=1)` on CPU succeeds.
- [ ] **Step 2: Run tests.** Expected: FAIL.
- [ ] **Step 3: Implement**.
- [ ] **Step 4: Run tests.** Expected: 2 passed. Then run the smoke test for every available adapter × supported mode. Record the chosen dataset name in the report as **declared pilot-exposed**.
- [ ] **Step 5: Commit** `"Add smoke run and records"`.

### Task 12: Training-free arms and TabPFN limits

**Files:** Create `tor/baselines.py`, `tor/tabpfn_arms.py`, `tests/test_baselines.py`

**Interfaces:**
- Consumes: `probe_grid` (Task 10), `measure`, `append`.
- Produces: `TRAINING_FREE = {"minirocket": ..., "mr_hydra": ..., "quant": ...}`, each a factory returning an aeon/sklearn pipeline (MiniRocket + `RidgeClassifierCV(alphas=np.logspace(-3,3,10))`; MultiRocket and Hydra features concatenated with the same ridge; QUANT + `ExtraTreesClassifier(n_estimators=200, max_features=0.1, criterion="entropy")` as in aeon's QUANT defaults, to be verified). `time_baseline(name, shape: ProbeShape, n_series=512) -> RunRecord` (fit + transform time on synthetic data, CPU only, `n_jobs=16`). `tabpfn_limits() -> dict` (version, max samples/features/classes per the installed package and docs, whether a GPU is used, local vs hosted). `rocketpfn_pipeline()` and `masht_pipeline()`, built from each paper's description of its feature reduction or ensembling step, cited in docstrings; any detail the paper doesn't give is recorded in `extra["unspecified"]`. `time_tabpfn(recipe: Literal["rocketpfn","masht"], shape) -> RunRecord` (hosted runs use `tabpfn-client`, read `TABPFN_TOKEN`, and set `hosted=True`). `datasets_exceeding_limits(meta, limits) -> list[str]`.
- [ ] **Step 1: Write failing tests**:
  - `test_baselines_fit_on_synthetic`: each of the 3 factories fits and predicts on `randn(40,1,64)` with 2 classes without error (prediction output is used only for its shape).
  - `test_exceeding_limits`: a fixture where one row has `classes=12` and `limits["max_classes"]=10` returns that row's name.
  - `test_hosted_run_flags_and_needs_token`: with `TABPFN_TOKEN` unset, `time_tabpfn("masht", ...)` with hosted mode returns `status=="not_run: no TabPFN token"`. With the client mocked, `cost["hosted"] is True`.
- [ ] **Step 2: Run tests.** Expected: FAIL.
- [ ] **Step 3: Implement**. Install `tabpfn` and `tabpfn-client`, then refresh the lock file.
- [ ] **Step 4: Run tests**, then time every baseline and both TabPFN recipes on the `FITS_SHAPE` and p50 shapes. Write the `tabpfn_limits` output as a `kind="tabpfn_limits"` record.
- [ ] **Step 5: Commit** `"Add training-free arms, TabPFN recipes and limits"`.

### Task 13: Literature and HC2 availability (research, not code)

**Files:** Create `reports/phase0/literature.md`, `reports/phase0/hc2_availability.md`, `tor/hc2.py`, `tests/test_hc2.py`

**Interfaces:**
- Produces: `published_results_coverage(meta, classifier: Literal["HC2","MR-Hydra"]) -> DataFrame` (columns `name, available: bool, resamples: int | None, source_url`), from the tsml-eval / timeseriesclassification.com published result files (verify the location).
- [ ] **Step 1: Write the failing test** `test_coverage_columns`: on a mocked results CSV with 2 of 3 datasets present, the function returns 3 rows, 2 of them `available`.
- [ ] **Step 2: Implement and run.** Write coverage for HC2 and MR-Hydra on all UCR/UEA rows into `hc2_availability.md`. State that these are full-label results only, and say whether resample 0 equals the official split.
- [ ] **Step 3: Literature.** Run three parallel research subagents: (a) fine-tuned TSFM vs ROCKET-family classification since 2026-09 on arXiv, Semantic Scholar, OpenReview and the 2026 workshops; (b) full-text reads of the three unread papers in brief §5.3; (c) recheck the R2R tables and UniShape code/weights. Each must cite full text, not abstracts, and map each finding to brief Q1–Q6. Merge the results into `literature.md`, marking anything unverified as such.
- [ ] **Step 4: Commit** `"Add Phase 0 literature and HC2 availability"`.

### Task 14: Phase 0 report

**Files:** Create `tor/report.py`, `tests/test_report.py`, `reports/phase0.md` (generated)

**Interfaces:**
- Consumes: `read_all` (Task 2), the availability CSV (Task 8), `FITS_SHAPE` (Task 10), and the markdown files from Task 13.
- Produces: `model_table(records, availability) -> DataFrame` with columns `model, family, available, licence, params_total, fits_full, fits_lora, fits_head, ckpt_needed, s_per_epoch_fits (full), shared_flag, hosted, notes`. `fits_*` is `yes`, `no (oom)`, or `missing: <reason>`. `s_per_epoch` comes only from the `FITS_SHAPE` records; when both shared and unshared records exist, the unshared one is preferred, and the flag column shows which was used. Also `render_report() -> str`.
- [ ] **Step 1: Write failing tests**:
  - `test_missing_shown_not_imputed`: a model with no `FITS_SHAPE` record gives `fits_full == "missing: no probe record"`.
  - `test_unshared_preferred`: two records, one shared at 10 s and one unshared at 8 s, give 8 with `shared_flag False`.
  - `test_report_has_no_score_columns`: the rendered markdown doesn't contain "accuracy", "F1" or "val_".
- [ ] **Step 2: Run tests.** Expected: FAIL.
- [ ] **Step 3: Implement**. The report sections follow the brief's format: what was run, the model table, the full probe grid as an appendix CSV link, the training-free timing, the TabPFN limits, HC2 availability, the literature, deviations (including the `TOR_CACHE` location change and any `not_run`), the declared pilot-exposed smoke dataset, anything unexpected, and next steps.
- [ ] **Step 4: Run tests**, then `python -m tor.report > reports/phase0.md`. Read it end to end and check that every number traces to a run record.
- [ ] **Step 5: Commit** `"Add Phase 0 report"`.
