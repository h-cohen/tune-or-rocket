# Task 9d report: timesfm25
Package: `timesfm` 3.0.2 (PyPI wheel; contains timesfm_2p5 code), installed --no-deps (torch/transformers/peft untouched). Adapter uses TimesFM_2p5_200M_torch_module directly (wrapper's load_checkpoint moves to CUDA + torch.compile).
- Params: total 231,289,280 (with forecast heads; model name says 200M, discrepancy not investigated); unused by classifier (output_projection_point + _quantiles) 32,768,000. Trainable (3 cls, 1 ch): head 3,843; lora 1,232,643; full 198,525,123.
- Modes: head, lora, full, scratch all supported. lora_targets: qkv_proj (fused q/k/v), out -> stacked_xf.N.attn.{qkv_proj,out}, N=0..19. Grad ckpt: yes (checkpoint per stacked_xf layer).
- Embedding: replicates package decode prefill (running stats, revin, masks, tokenizer, stacked_xf); last layer output mean-pooled over real patches (fully padded patches excluded); per channel, d=1280*C. Mean pooling = harness choice (causal attention; no package embedding path).
- Length: truncate last 16384 (context_limit); left-pad to multiple of 32 with zeros + mask (forecast_naive). preprocess rejects NaN/inf.
- Recipe: harness default (no published classification recipe found in paper/repo/card; not exhaustive). Corpora: cited paper 2310.10688 list; 2.5 corpus not documented; UCR/UEA not documented.
- Scratch: package-default init (module has no custom init), seeded.
- Tests: tests/models/test_timesfm.py (5 tests: weights equal checkpoint tensor, scratch differs, param counts, lora names, preprocess, hidden states exactly equal package decode prefill via hook, padded-patch exclusion, channel independence). Contract test passes. Full suite: 100 passed. No separate RED run captured (tests written alongside the adapter).
- Commit: "Add TimesFM adapter" (no trailers, per common rules).

## Fix round 1
Docstring/test: size gap explained as measured subtraction (231,289,280 - 32,768,000 = 198,521,280, matches "200M"; card does not itemise); test asserts (total - unused) within 1% of 200M, total bound labelled measured. Docstring note that pad never yields a fully padded patch. New test: length-5 and constant inputs give finite embeddings.
Command: pytest tests/models/test_timesfm.py -> 6 passed.
