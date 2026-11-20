# Task 9c report: Chronos adapters
Commit: "Add Chronos adapter". Files: tor/models/chronos.py, tests/models/test_chronos.py, tor/models/__init__.py, requirements.lock.
Install: chronos-forecasting 2.3.2 installed normally (torch<3, transformers<6 compatible; torch 2.14.1, transformers 5.23.0, peft 0.22.0 unchanged).

| | chronos-bolt-small | chronos-bolt-base | chronos2 |
|---|---|---|---|
| total params (loaded, incl. decoder/forecast head) | 47,718,016 | 205,292,928 | 119,477,664 |
| removed from classifier (unused) | 27,702,144 | 117,850,368 | 3,653,280 |
| trainable (3 cls, 1 ch) head/lora/full | 1,539 / 198,147 / 20,017,411 | 2,307 / 592,131 / 87,444,867 | 2,307 / 1,181,955 / 115,826,691 |
| modes | head, lora, full, scratch | same | same |
| lora_targets | q,k,v,o (encoder.block.N.layer.0.SelfAttention.*) | same | q,k,v,o (encoder.block.N.layer.{0 time,1 group}.self_attention.*) |
| grad ckpt | yes (torch.utils.checkpoint per encoder block, train mode) | yes | yes |
| max_length | 2048 | 2048 | 8192 |
| multivariate | per_channel | per_channel | native |

- Embedding: Bolt = `model.encode()` encoder tokens (incl. [REG]) mean-pooled, channels concatenated. Chronos-2 = `Model.encode(context=(n*C,L), group_ids=sample idx)` (as Chronos2Pipeline.embed), mean over context patches + [REG], excluding the masked future-output placeholder; channels concatenated (d=768*C).
- Length: both packages truncate to last context_length (cited in docstring); preprocess truncates to last max_length, models NaN-pad to patch multiple themselves. preprocess raises on NaN input.
- Weights verified: test compares state_dict q.weight with model.safetensors tensor; scratch differs, seeded reproducible. Scratch = AutoConfig + Model(config).
- Recipe: harness default (lr 1e-4, wd 0.01, 100 ep, bs 32). No classification recipe found in the two papers/package; Chronos2Pipeline.fit is forecasting (lr 1e-6, 1000 steps) -- search limited, not exhaustive.
- Corpora: Bolt card: "nearly 100B observations", corpora not listed; Chronos-corpus reuse inferred, not verified; UCR/UEA not documented. Chronos-2 card: chronos_datasets subset, GiftEvalPretrain subset, synthetic; UCR/UEA not documented.
- Jain et al. 2611.06988 exists ("Big Models, Small Gains: Frozen Forecasters as Classifiers on MONSTER"); abstract checked, encoder-only/mean-pool detail NOT verified in text.

Tests: tests/models/test_chronos.py 7 passed; contract -k chronos 3 passed; full suite 92 passed.
TDD note: tests were written together with the implementation and passed on first run; no separate RED run was recorded (honest deviation).
Concerns: none blocking. Chronos-2 LoRA also adapts group attention (documented).

## Fix round 1
Pooling changed per ruling: mean over PATCH tokens only, excluding [REG] (Bolt) and [REG] + future-output token (Chronos-2). `_Backbone.tokens()` returns hidden states and n_patch; forward pools hidden[:, :n_patch]. Docstring now cites Jain et al. App. B (patch tokens without [REG]), Sec. 3 (mean and max reported; harness uses mean), Bolt-base only; Chronos-2 same rule by harness choice; the "not verified" hedge removed. New test `test_chronos_pooling_is_patch_tokens_only` (bolt-small, chronos2): hidden states equal the package's own encode() output, n_patch/extra token counts, embed == mean over patch tokens, differs from all-token mean.
Command: pytest tests/models/test_chronos.py tests/models/test_contract.py -m slow -k chronos -> 12 passed.
