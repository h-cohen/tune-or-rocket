# Task 9h: NuTime adapter (nutime)
Commit: "Add NuTime adapter". Files: tor/models/nutime.py, tests/models/test_nutime.py, tor/models/__init__.py (module list). No packages added (requirements.lock unchanged).
- Source: pinned clone <TOR_CACHE>/nutime/NuTime @ 9acdd3d0e82df0914217ef752fa3b95ec82cd960 (MIT, nothing vendored); ckpt/checkpoint_bias9.pth sha256 65543d55...f7a2 verified; is_available() gives (False, reason).
- Imports: temporary `src*` packages + timm stub (trunc_normal_, DropPath raising) removed from sys.modules after import (test checks no leakage).
- Params (measured): their enc+WinT with 2-class fc = 2,395,586 (encoder 27,968; WinT 2,367,618); classifier backbone 2,395,328; unused = their fc (258 for 2 classes). Checkpoint tensors total 10,321,044 (incl. BYOL momentum copy, predictor, 4096-d projector, pos_embed/scales buffers) of which backbone used is 2,395,328 params. Trainable (3 classes, 1 ch): head 387, lora 110,979, full 2,395,715.
- Modes: head, lora, full, scratch all supported. grad ckpt supported (checkpoint on PreNorm blocks). lora_targets: to_qkv, to_out.0 (6 layers, r8 a16).
- Length: all series linearly resized to 512 (TSResize, transforms.py:458/179; WNE needs len%16==0); 512 = train_ssl.json/train_ft.json value; demo's 176 is Epilepsy-specific. max_length=512. NaN/inf -> ValueError (theirs zero-fills).
- Channels: per_channel, d=128*C (checkpoint is univariate; their channel_embedding is not pretrained).
- Embedding: WinT.features (LayerNorm'd cls token), exact torch.equal vs their nn.Sequential(encoder, WinT) at lengths 512 and 176. Resize also torch.equal vs their TSResize.
- Head Linear(128*C,K); head mode uses their freeze_backbone init N(0,0.01)/0 (cls_exp.py:59-72).
- Recipe: demo_ft_epilepsy.json: AdamW lr 2e-4, wd 0.05, 100 epochs, bs 2048, cosine, warmup 0.1. Dataset-specific (bs 2048 > most UCR train sets = full-batch); train_ft.json differs (lr 2e-3, bs 1024, 200 ep).
- Pretraining: 1.89M univariate series, UCR/UEA TRAIN splits included (per lit-c notes; arXiv abstract itself only says "over a million sequences"; config dataset "ucr-merged"). Title/TMLR 2024 verified from arXiv page comments.
- TDD: RED `pytest tests/models/test_nutime.py` -> ModuleNotFoundError tor.models.nutime. GREEN: 11 passed. Contract `-k nutime -m slow`: 1 passed. Full suite: 142 passed (510s).
- Concerns: head mode backbone stays in train mode (dropout 0.1) if trainer calls .train(); same as other adapters.

## Fix round 1
Only tor/models/nutime.py changed (docstring, `pretraining_corpora`, `recipe_source`).
1. pretraining_corpora: now cites configs/train_ssl.json dataset "ucr-merged" (UCR inclusion likely); 1.89M size, UEA inclusion and train-only splits attributed to notes lit-c.md, marked NOT verified against paper/repo data code.
2. recipe wording: their schedule is cosine with 10 warmup epochs; harness trainer (tor/train.py) applies linear warmup only, no cosine decay (docstring and recipe_source).
Command: `pytest tests/models/test_nutime.py -q` -> 11 passed.
