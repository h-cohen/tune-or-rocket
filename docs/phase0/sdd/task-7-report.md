# Task 7 report
Implemented tor/models/{__init__,base,inceptiontime}.py, tor/train.py, tests/test_train.py, tests/models/test_contract.py; pyproject: slow marker, tor.models package.
RED: pytest tests/test_train.py tests/models -> collection errors (modules missing).
GREEN: pytest -m "slow or not slow" -> 54 passed.
Notes: InceptionTime.embed returns GAP features of the current net (needs load(("random",seed))). Channel count taken from last preprocess call. Divergence test uses inputs scaled by 1e35 since AdamW steps are bounded by lr. No eval code in tor.train.

## Fix round 1
All 6 findings fixed: contract test count_params order and frac==1.0; classifier(n_classes, mode, n_channels, seq_len, grad_ckpt, seed); AdapterUnavailable/is_available/get_adapter/duplicate-register, only ImportError/AdapterUnavailable caught; supports_grad_ckpt checked in _check_mode; train_epochs puts fully frozen submodules in eval (test with BN+dropout); InceptionTime builds under fork_rng, no self._net side effect in classifier, embed in eval (net built lazily from load seed).
Command: pytest -m "slow or not slow" -> see below.
57 passed, 1 warning in 11.26s
