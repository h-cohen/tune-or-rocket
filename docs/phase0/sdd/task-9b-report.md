# Task 9b report: MOMENT adapters

Commit 5214e97 "Add MOMENT adapter". Files: tor/models/moment.py, tests/models/test_moment.py, tor/models/__init__.py (module list), requirements.lock (+momentfm==0.1.4).

## Install
`pip install momentfm --no-deps` (its metadata pins transformers==4.33.3, huggingface-hub==0.24.0, numpy==1.25.2, which would downgrade the env). The package imports and runs unchanged on transformers 5.23 / hub 2.4 / torch 2.14. No other deps needed. torch/transformers/peft untouched.

## Verification
- Adapter logits equal the official `MOMENTPipeline(task_name="classification", reduction="concat")` output exactly (max diff 0.0) with the same head weights, on a left-padded, masked input (2 channels).
- Contract: `pytest tests/models/test_contract.py -k moment -m slow` -> 3 passed in 119 s total (large included; no xlarge marker needed). Covers head/lora/full/scratch, grad_ckpt forward+backward, (4,1,128) and (4,3,128).
- tests/models/test_moment.py: 5 passed (params, scratch != pretrained and seed-determinism, lora count between head and full, head backbone fully frozen, pad/sub-sample, embed shape, mask independence across batch).
- Full suite: 81 passed. (RED step: the test file was written against the finished module in the same pass; no separate failing-run recorded.)

## Per adapter (identical except size)
| | moment-small | moment-base | moment-large |
|---|---|---|---|
| hf_repo | AutonLab/MOMENT-1-small | -base | -large |
| d_model | 512 | 768 | 1024 |
| params, backbone (no head) | 35,337,408 | 109,635,456 | 341,240,320 |
| of which unused T5 embed_tokens | 16,449,536 | 24,674,304 | 32,899,072 |
| trainable head/lora/full (3 cls, 1 ch) | 1,539 / 230,915 / 35,338,947 | 2,307 / 592,131 / 109,637,763 | 3,075 / 1,575,939 / 341,243,395 |

Paper (Tab. 8) quotes 40M / 125M / 385M; measured counts are lower (large: 341M vs 385M). Not reconciled (paper may count differently).

- Modes: head, lora, full, scratch (all supported).
- lora_targets: `q`, `k`, `v`, `o` (encoder.block.N.layer.0.SelfAttention.*); r=8, alpha=16. Patch embedder frozen in head and lora.
- Grad ckpt: supported (HF T5Stack `activation_checkpointing_enable`); contract passes. momentfm defaults it ON; adapter sets it explicitly (off unless requested).
- Length: fixed 512 (patch 8). Shorter: left-pad zeros + input mask (paper Sec. 3.2; momentfm/data/classification_dataset.py). Longer: uniform sub-sample to 512 via round(linspace(0,L-1,512)); the paper says only "sub-sample", package code has no routine (its dataset would error), so the exact index rule is a harness choice.
- Mask transport: preprocess marks the pad with NaN; model derives mask = ~isnan(x[:,0]) and zero-fills. Real NaNs in data would be read as padding (inputs must be imputed).
- Channels: per channel, `reduction="concat"` (MOMENT.classify; repo tutorial default), head sized from n_channels. Embedding = mean over patches of the concat features (ClassificationHead input), d*C.
- Scratch: `MOMENT(config.json)` with seeded RNG; T5EncoderModel(config) random HF init + package default patch-embedding init. No pretrained weights.
- Recipe: Recipe(lr=1e-4, wd=0.0, epochs=5, batch=16, warmup_frac=0.1). Source: momentfm repo tutorials/finetune_demo/classification.py (Adam, OneCycle 1e-6 -> 1e-4, batch 16, 5 epochs, concat). The paper has NO classification fine-tuning recipe (App. F.2 evaluates frozen embeddings + SVM); its general fine-tuning text gives batch 64 and one-cycle schedule for Large, but the peak lr values were lost in the HTML text extraction. Deviations: no one-cycle decay, AdamW wd 0 for Adam.
- Pretraining corpora (arXiv 2402.03885v3, Sec. 3.1, App. E): Time Series Pile, 13M series / 1.23B timestamps from Informer forecasting sets, Monash archive, UCR/UEA classification archive (159 datasets), TSB-UAD. UCR/UEA IS included, TRAIN splits only (paper: only training splits seen in pre-training; creator splits, else 60/10/30 with seed 13). Which individual UCR/UEA datasets entered the Pile is not listed. For the study: UCR/UEA test data is not seen, but train data is.

## Concerns
1. MOMENT's own classification head mean-pools over all 64 patches including fully padded ones (mask ignored). For short series (e.g. L=128: 48/64 patches padded) the pooled embedding is dominated by pad-patch outputs. Reproduced faithfully (matches official path); a masked mean would be a deviation. Flag for the Phase 0 report.
2. NaN-as-pad transport means any harness step that touches preprocessed tensors (e.g. on-the-fly normalisation) must preserve NaNs.
3. Paper classification experiments only covered UCR sets shorter than 512; longer-series handling is the harness's.
4. Unused embed_tokens inflates "full" trainable param counts by 16-33M (no gradient).
5. `xlarge` marker not needed, so pyproject untouched.

## Fix round 1
1. Unused embed_tokens: instead of requires_grad=False (which would make full/scratch trainable/total < 1 and break the contract test's `fracs == 1.0` assertion in test_contract.py, out of scope), `classifier()` sets `encoder.embed_tokens = None` in the returned model (never called; inputs_embeds are fed). Classifier counts are therefore effective counts, total == trainable for full/scratch. Added `unused_param_count()`. Docstring corrected: measured totals 35.3/109.6/341.2M include unused embeddings; paper sizes differ, gap unexplained. Effective trainable (3 cls, 1 ch) full: small 18,889,411; base 84,963,459; large 308,344,323 (head/lora unchanged: 1,539/230,915; 2,307/592,131; 3,075/1,575,939). Test asserts full == total - unused + head and total == trainable.
2. `preprocess` raises ValueError on NaN input; tested.
3. Mask test replaced by test_moment_mask_is_used_and_embed_shape (masked vs zero-padded-unmasked embeddings differ; batch independence; eval mode) plus test_moment_preprocess_rejects_nan.
Tests: test_moment.py 6 passed; moment contract (small/base/large) 3 passed (106 s).
