# Task 9g: UniShape adapter

Commit: "Add UniShape adapter". Files: tor/models/unishape.py, tests/models/test_unishape.py, tor/models/__init__.py (module list). requirements.lock unchanged (no packages left installed: timm/fastai and deps were tried, uninstalled; fastai.basics and timm trunc_normal_ are stubbed in the adapter).

Licence: docstring carries the required no-licence statement. Clone at <TOR_CACHE>/unishape/UniShape pinned to 78d326590ee959bd473c4b07ad6cbc13977c8625; checkpoint sha256 1a908b6b...65df verified before torch.load. Nothing vendored. is_available() returns (False, reason) if clone missing / wrong commit / ckpt missing.

Findings vs brief:
- unishape_finetune.py does NOT resize to 512 (series_size = native length). 512 resize (TSResize) is pretrain/zeroshot only. Adapter: native length, max_length=None, series <16 linearly resized to 16.
- Fine-tune forward uses a single window scale (scale_len, per-dataset table); harness uses script default 5 (window 4).
- Univariate only in their script: per_channel, d=128*C. Embedding = their transformer_enc class token. Head = their fc MLP (Linear-BN-ReLU-Dropout-Linear), fresh init.
- Recipe (unishape_finetune.py): AdamW lr 1e-3, wd 0.05, 300 epochs, batch 512 default, constant LR. Prototype loss, 5-model ensemble not reproduced.
- Params: their fine-tune model (2-class fc) 3,133,203 vs paper 3.1M (matches). Unused by classifier 565,634 (4 unused unit_scale_list scales, whole unit_scale_list_finetune, fc, fc_token_shape, layer_norm, class_proto_centers, cls_token) => backbone 2,567,569. Trainable (3 classes, 1 ch): head 34,307; lora 144,899; full 2,601,876.
- Modes: head, lora, full, scratch. lora_targets: to_qkv, to_out.0 (6 layers, 24 lora tensors). grad ckpt supported (checkpoint around PreNorm blocks), contract test passes.
- Pretraining: 1.89M samples incl. UCR/UEA TRAIN splits + 8 TFC datasets.

Tests: tests/models/test_unishape.py 8 passed (exact torch.equal vs their forward at L=100 and 40; ckpt tensor equality; scratch/seed; lora names; short/constant; NaN; availability). Full suite: 128 passed (incl. contract test for unishape). TDD note: implementation was written before the tests (no RED run recorded).
Concern: no licence; user decision on inclusion.

## Fix round 1
1. fastai/timm stubs are now marked and removed from sys.modules right after the UniShape import (their modules keep references); find_spec("timm"/"fastai") no longer raises. Test: test_unishape_stubs_not_left_in_sys_modules.
2. _check_clone now hashes the checkpoint (cached per process, lru_cache) so is_available() is False on mismatch. Test: test_unishape_hash_mismatch_unavailable (monkeypatched expected hash). (Checkpoint is 56 MB, not 12.)
3. "timm trunc_normal_ is torch's" marked inferred/not verified in docstrings.
Result: test_unishape.py 10 passed; full suite 130 passed.
