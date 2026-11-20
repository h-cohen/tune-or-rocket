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

