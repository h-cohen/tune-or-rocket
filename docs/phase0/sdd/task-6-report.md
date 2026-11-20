# Task 6 report
Implemented tor/splits.py (BUDGETS, DRAWS, draw_seed, support_draw, InsufficientClassError, budget_key, save_draws) and tests/test_splits.py (8 tests).
RED: `python -m pytest tests/test_splits.py` -> ImportError: cannot import name 'splits' from 'tor'.
GREEN: full suite `python -m pytest -q` -> 49 passed.
Notes: fraction budget = min(n, max(1, round(f*n))) (banker's rounding); fraction 1.0 -> arange(len(y)); classes iterated in sorted label order; save_draws stores {"error": msg} per failing budget; tested on synthetic y only, no real split files generated.
