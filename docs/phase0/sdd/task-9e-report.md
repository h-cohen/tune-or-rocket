# Task 9e: Toto adapter report
Commit: "Add Toto adapter". Files: tor/models/toto.py, tests/models/test_toto.py, tor/models/__init__.py (module list), requirements.lock.
Install: `pip install --no-deps toto-ts` (0.2.0; pins torch==2.7.0, transformers==4.52.1, numpy==1.26.4 would downgrade) plus gluonts 0.17.0, lightning 2.6.6 (+deps), jaxtyping, rotary-embedding-torch. torch/transformers/peft untouched. pip warns toto-ts wants tqdm==4.66.3/transformers==4.52.1 (not enforced, adapter works).
Package `Toto.from_pretrained` is broken with huggingface_hub 2.x (missing proxies/resume_download kwargs); adapter uses `Toto.load_from_checkpoint(snapshot_dir)` instead.
- total params 151,306,080 (paper: 151M); unused by classifier (unembed + Student-T head) 37,871,712; backbone used 113,434,368
- trainable (3 classes, 2 ch): head 4,611, lora 446,979, full 113,438,979
- modes: head, lora, full, scratch (all supported); grad ckpt supported (per transformer layer, non-reentrant)
- lora_targets: wQKV, wO (layers 0-10 time-wise, layer 11 space-wise; 48 lora tensors)
- native multivariate: channels fed as variates (id_mask ones); pooled over real patches per variate, concatenated: d = 768*C
- embedding: package scaler (CausalPatchStdMeanScaler) -> patch_embed -> transformer output (before unembed/distribution); exact torch.equal vs package TotoBackbone.backbone transformer output (hook)
- length: left-pad to multiple of 64 like pad_array/pad_id_mask; max_length 4096 (paper "L=4096", README example; code has no hard limit; pretraining context not verified); longer: keep last 4096; NaN/inf -> ValueError
- recipe: harness default (repo has only forecasting finetune pipeline)
- corpora: paper 2.36T points, 43% Datadog observability, GIFT-Eval Pretrain + Chronos collections (minus GIFT-Eval overlap), synthetic; UCR/UEA not documented (only ~half of the paper HTML read)
- scratch: Toto(**config.json) seeded, PyTorch default init; classifier deepcopy impossible (gluonts scaler not picklable) so classifier rebuilds from config + load_state_dict(strict)
Tests: tests/models/test_toto.py 7 passed; contract test toto passed; full suite 109 passed.
TDD note: tests written together with the implementation; only RED seen was the deepcopy TypeError (fixed). Not a strict RED-first run.
Concerns: padded patches are not masked in time attention (package behaviour); no GPU used; Toto-2.0 family not covered.
