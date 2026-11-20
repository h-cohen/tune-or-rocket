# Task 9f Reverso report
Commit affdedd "Add Reverso adapter". Files: tor/models/reverso.py, tests/models/test_reverso.py, tor/models/__init__.py, requirements.lock.
Tests: contract (reverso, reverso-small; head/lora/full/scratch, grad ckpt) 2 passed; test_reverso.py 8 passed; non-slow suite 64 passed.
Install: pip --no-deps git+https://github.com/shinfxh/reverso@35f724d (MIT), pinned in lock. Uses torch-native `reverso_torch` (no FlashFFTConv/fla).
Params: reverso-base 2,609,585 (+268,288 PE buffer = 2,877,873 ckpt values); unused decoder 164,529; used 2,445,056. small 550,161; unused 110,897; used 439,264.
Trainable (3 cls, 1 ch): reverso head 1283 / lora 34051 / full 2446339; small head 643 / lora 8835 / full 439907.
lora_targets: attention.{q_proj,k_proj,v_proj,o_proj} of DeltaNet (no softmax attention; conv mixers have none). Grad ckpt: yes.
Embed: Jain 2611.06988 (via summarising fetch): all layers' hidden states, mean over real positions (8x128 / 4x64 per channel). Classifier head: last layer mean+max, LayerNorm, zero-init Linear. Length: 2048; truncate last 2048, left-pad first value in backbone.
Recipe: Jain AdamW lr1e-4 (head 1e-3), wd .01, cosine, clip 1, ~64/update; epochs=100 harness (full-label epochs unstated); batch 64.
Concerns: reverso_torch.Model ignores use_output_pe (adapter adds post_pe_q_proj); pure-torch DeltaNet is a 2048-step Python loop (slow); 128-length inputs padded to 2048; reverso-base mapping inferred (HF config == GitHub checkpoints/reverso/args.json; weights not byte-compared, LFS not fetched); Jain details unverified against raw HTML.
