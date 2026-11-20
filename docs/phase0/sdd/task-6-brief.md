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

