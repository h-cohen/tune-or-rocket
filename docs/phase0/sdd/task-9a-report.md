# Task 9a report: Mantis adapters (commit 29d20b4 "Add Mantis adapter")

Package: mantis-tsfm 1.1.0 (PyPI, repo vfeofanov/mantis). Installed without touching torch/peft/transformers. It pulled datasets 5.2.0, einops, pyarrow, and DOWNGRADED pandas 3.0.6 -> 2.3.3 (flagged; full suite 73 passed afterwards). requirements.lock regenerated.

Files: tor/models/mantis.py, tests/models/test_mantis.py, requirements.lock, and one line in tor/models/__init__.py (added "tor.models.mantis" to _ADAPTER_MODULES; outside the stated scope but required for registration).

## Per adapter
| | mantis8m | mantisv2 |
|---|---|---|
| class / repo | mantis.architecture.MantisV1 / paris-noah/Mantis-8M | MantisV2 / paris-noah/MantisV2 (different class, needed) |
| backbone params | 8,103,936 | 4,188,672 |
| trainable (3 cls, 1 ch) head / lora / full | 1,283 / 222,467 / 8,105,219 | 1,283 / 75,011 / 4,189,955 |
| modes | head, lora, full, scratch | same |
| lora_targets | to_qkv, to_out.0 | wQKV, wO |
| grad ckpt | no (not implemented; models small) | no |
| max_length | 512: preprocess linearly interpolates any length to 512 (README "Prepare Data"; V1 fixes seq_len=512 in config, V2 accepts multiples of 32 but same resize used) | same |
| channels | per channel, (n*C,1,512) -> 256-d cls token, concatenated, d=256*C | same |
| scratch | package class with default ctor args (= repo config.json), seeded random init | same |
| recipe | AdamW lr 2e-4, wd 0.05, 500 ep, batch 128, warmup_frac 0.02 (10 epochs, from package trainer; harness has no cosine). Source: arXiv 2502.15637v2 App. C.1 (full fine-tuning, last-epoch result, head = LayerNorm + Linear) | same |
| pretraining | ~1.89M real series (NuTime-like corpus); paper App. B.1 says this corpus includes UCR and UEA train sets | synthetic CauKer data only (Sec 3.3), no UCR/UEA |

Head: LayerNorm+Linear per paper (package default is BatchNorm1d+Linear).
Note: deepcopy of MantisV2 fails (rope staticmethod), so pretrained state_dict is cached and loaded into fresh instances.
Mantis-8M pretraining claim for v1 is inferred from v2 paper text ("similar to the one used for the first version of Mantis (1.89M)"), worth a check against the v1 paper.

## TDD
RED: `pytest tests/models/test_mantis.py -q` -> 6 failed (adapters not registered). GREEN: 6 passed (prints counts above).
Contract: `pytest tests/models/test_contract.py -q -k mantis` -> 2 passed (CPU, no GPU used). Full suite: 73 passed.

## Fix round 1
- mantis8m pretraining_corpora now VERIFIED from v1 paper Sec. 2.4 (union incl. UCR, UEA minus EigenWorms/InsectWingbeat, 7M series; eval test sets excluded; split of UCR/UEA unspecified). Discrepancy with v2 text (1.89M) noted, not reconciled. mantisv2 cites v2 Sec. 3.3.
- recipe_source/comment now state deviations: constant lr after warmup vs cosine; batch 128 (paper) vs package default 256.
- Tests added: head-mode backbone has 0 trainable params; embed() leaves backbone in eval. `pytest tests/models -q` result: see below.
