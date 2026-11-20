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

