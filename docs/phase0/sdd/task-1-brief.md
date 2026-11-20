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

