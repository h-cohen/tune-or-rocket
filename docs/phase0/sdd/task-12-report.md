# Task 12a report: training-free arms and TabPFN limits (code + tests)

## Implemented
- tor/baselines.py: TRAINING_FREE factories (minirocket, mr_hydra, quant; threads arg), time_baseline, datasets_exceeding_limits, run_arms (config_hash resumable, only status ok counts as done), CLI `python -m tor.baselines --which ... [--shapes fits,p50] [--hosted] [--threads N] [--n-series] [--masht-budget] [--timeout] [--force] [--runlog]`. Cells run through memtime.isolated_runner (spawned child, timeout): CPU arms and hosted use device "cpu"; local TabPFN lets the child lease a GPU.
- tor/tabpfn_arms.py: time_arm core (fit_s, transform_s, predict_s, per-1000 variants, n_features, threads, cost), RocketPFNArm/rocketpfn_pipeline, MashtArm/masht_pipeline, masht_budget, time_tabpfn (hosted flag, token, mockable _make_classifier/_set_hosted_token), tabpfn_limits.
- tests/test_baselines.py (10 tests, ~7 s alone). requirements.lock refreshed.

## aeon verification (aeon 1.6.0 installed source; cited in tor/baselines.py docstring)
- MiniRocket: MiniRocketClassifier._fit = MiniRocket(10000, 32) -> StandardScaler(with_mean=False) -> RidgeClassifierCV(logspace(-3,3,10)). Same here.
- MR-Hydra: MultiRocketHydraClassifier._fit = HydraTransformer(8,64)+_SparseScaler, MultiRocket()+StandardScaler, hydra block first, RidgeClassifierCV. Same here (MRHydraFeatures).
- QUANT: QUANTClassifier._fit = QUANTTransformer(6,4) + ExtraTreesClassifier(200, max_features=0.1, criterion="entropy", class_weight None). Deviation: n_jobs=threads, random_state=0.

## RocketPFN (arXiv 2606.21786, read via HTML)
Sec.3: G=10 groups x 1000 standard-Rocket kernels -> 2000 features each, no reduction, one TabPFN per group, per-instance z-norm (aeon Rocket normalise=True); Eq.1 mean proba + argmax; Sec.2 e=8 estimators, full train set as context; v2.5; >10 classes not handled (excluded). Unspecified (in extra["unspecified"]): checkpoint/device/hparams beyond e=8, seeds, sequential vs parallel groups, aeon options.
## MASHT (arXiv 2607.19234v1)
Sec.3: MultiRocket (raw+diff) + Hydra (k=8), budget by total instances (<1000: 10000; <100000: 2000; else 200), split equally, concatenated, no scaling/reduction/ensembling; App.A TabPFN-3, low_memory, ignore_pretraining_limits=True. Unspecified recorded: MR kernel params (we use n_kernels=budget_MR//8), d formula mismatch (aeon d used), Hydra max_num_channels, standardisation (none), multivariate handling, >10 classes, row subsampling, seeds, n_estimators (extracted text lists "88", flagged uncertain; hosted max is 8, we use 8 and record the deviation).
Caveat: paper content was read through the WebFetch summarizer, not raw text; the 88 and the 200-feature tier are exactly the items it flagged for verification.

## Limits (tabpfn 10.2.0, tabpfn-client 0.10.1)
- local v2.5 (InferenceConfig.get_default): samples 50000, features 2000, classes 10, CPU cap 1000 samples (cpu_sample_limit). v2: 10000/500/10. v3: config lives in checkpoint, None with reason. Local raises beyond max classes (validation.validate_num_classes).
- hosted (queried live with token via ServiceClient.get_settings, non-billable): v2.5 50000/2000/10; v3 1,000,000 rows / 2000 cols / 160 classes; v3.5 1.1M / 23000 cols / 160 classes (default hosted version v3.5). tabpfn_limits(query_hosted=True).
- Weights: v2.5 and v3 checkpoints are in the HF repos Prior-Labs/tabpfn_2_5 / tabpfn_3 (gated, licence acceptance); both downloaded and loaded successfully here (cached auth token at ~/.cache/tabpfn), CPU smoke only.
- Install: tabpfn 10.2.0 + tabpfn-client 0.10.1 (no torch/transformers change; sklearn 1.9.1->1.9.0, pyarrow 26->25, pydantic 2.14.4->2.14.3; pip check complaints are pre-existing toto-ts pins).

## TDD
RED: `pytest tests/test_baselines.py` -> collection error (ModuleNotFoundError tor.baselines). GREEN: 10 passed in 7 s after fixes. Full suite: 178 passed (529 s, machine busy).

## Findings / concerns for the controller
1. Hosted MASHT with the default budget fails client-side validation if budget 10000 is used with v3 (cols cap 2000, ignore_pretraining_limits does not lift it): a 16-series smoke gave "train columns (9568) exceeds 2000" (recorded error in a tmp log). With n_series=512 the budget tier is 2000 (n_total=2*n_series=1024 >= 1000) so ~1.9k columns fit. Use --masht-budget to override; v3.5 allows 23000.
2. Local CPU is very slow (rocketpfn 1 group, 60 series, 4 threads: predict 29 s; masht 60 series predict 186 s while machine loaded); the real local runs need the GPU path (CLI default: child leases a GPU).
3. I made one tiny real hosted call (16 synthetic series, rejected at validation, no upload) during CLI smoke, plus a get_settings query. Nothing hit the real run log; smoke used a scratchpad log.
4. MASHT n_kernels floor of 42 added because aeon MultiRocket divides by zero for tiny budgets.
5. mr_hydra transform stage is timed on a fresh set, fit_s includes training-feature transform; for TabPFN arms fit_s includes upload (hosted) and per-group Rocket transforms.
6. runs/runlog.jsonl untouched by me (shows modified from the controller's runs, not committed).

## Fix round 1
1. NO_WEIGHTS now only for typed failures anywhere in the exception chain (__cause__/__context__): tabpfn TabPFNLicenseError / TabPFNHuggingFaceGatedRepoError, huggingface_hub GatedRepoError / RepositoryNotFoundError / HfHubHTTPError with 401/403 (tabpfn wraps download failures in a plain RuntimeError, hence the chain walk). Generic RuntimeError mentioning huggingface/license is now "error". traceback_tail kept for not_run too.
2. MASHT config and extra carry n_estimators_paper=88, n_estimators_paper_note="possibly a typo for 8", n_estimators_used=8 (config key "n_estimators" renamed to n_estimators_used for all recipes).
3. Error message and traceback pass through tor.config.redact (covers HF_TOKEN and TABPFN_TOKEN via SECRET_KEYS) before entering the record.
Tests added: typed classification, secret scrubbing, estimator deviation. `pytest tests/test_baselines.py`: 13 passed.
