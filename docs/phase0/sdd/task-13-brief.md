### Task 13: Literature and HC2 availability (research, not code)

**Files:** Create `reports/phase0/literature.md`, `reports/phase0/hc2_availability.md`, `tor/hc2.py`, `tests/test_hc2.py`

**Interfaces:**
- Produces: `published_results_coverage(meta, classifier: Literal["HC2","MR-Hydra"]) -> DataFrame` (columns `name, available: bool, resamples: int | None, source_url`), from the tsml-eval / timeseriesclassification.com published result files (verify the location).
- [ ] **Step 1: Write the failing test** `test_coverage_columns`: on a mocked results CSV with 2 of 3 datasets present, the function returns 3 rows, 2 of them `available`.
- [ ] **Step 2: Implement and run.** Write coverage for HC2 and MR-Hydra on all UCR/UEA rows into `hc2_availability.md`. State that these are full-label results only, and say whether resample 0 equals the official split.
- [ ] **Step 3: Literature.** Run three parallel research subagents: (a) fine-tuned TSFM vs ROCKET-family classification since 2026-09 on arXiv, Semantic Scholar, OpenReview and the 2026 workshops; (b) full-text reads of the three unread papers in brief §5.3; (c) recheck the R2R tables and UniShape code/weights. Each must cite full text, not abstracts, and map each finding to brief Q1–Q6. Merge the results into `literature.md`, marking anything unverified as such.
- [ ] **Step 4: Commit** `"Add Phase 0 literature and HC2 availability"`.

