# Task 9 (FM adapters) — shared instructions for every family sub-task

Read the plan text for Task 9: task-9-brief.md (same dir). The interface it references has since been amended by controller rulings — the CURRENT interface (read tor/models/base.py, tor/models/__init__.py, tor/models/inceptiontime.py, tests/models/test_contract.py before writing code) is:
- `FMAdapter` class attrs: name, hf_repo, max_length, multivariate ("native"|"per_channel"), pretraining_corpora (cited), recipe_source, supported_modes, supports_grad_ckpt, lora_targets.
- `load(init)`, `preprocess(X) -> float32 (n,c,L')`, `embed(X) -> (n,d)`, `classifier(n_classes, mode, n_channels, seq_len, grad_ckpt=False, seed=0)`, `default_recipe()`, `lora_config()`, classmethod `is_available() -> (bool, str)`.
- Register with `@register`; registry skips only ImportError / AdapterUnavailable.
- Contract test (tests/models/test_contract.py, marker slow) is parametrized over available_adapters(); your adapter must pass it for (4,1,128) and (4,3,128).

Rules:
- `scratch` = same architecture with package-default random initialisation (load(("random", seed))), never pretrained weights. If the package cannot build a random-init model from config, build from the HF config (e.g. `from_config`) — document how.
- `head` mode: backbone frozen (requires_grad False), only the head trains. `lora`: peft LoRA on `lora_targets` (attention q/k/v/o projection module names of THAT model — inspect the module tree and list exact names), r=8, alpha=16, dropout 0; head trainable. `full`: everything trainable.
- Set `supports_grad_ckpt = True` only if you actually enable gradient checkpointing (e.g. HF `gradient_checkpointing_enable()` or torch.utils.checkpoint on blocks) and the contract test's grad_ckpt forward+backward passes.
- Embedding path: use the model's own classification/embedding path if it has one; otherwise encoder output mean-pooled over tokens. Per-channel models: process each channel independently, concatenate channel embeddings (d = d_model*C) — the head is sized from n_channels. Native multivariate models: document how channels are fed.
- Length: follow the model's own code for series longer than max_length / shorter than its minimum (resize, pad, truncate) — cite file/function in the docstring.
- `default_recipe()`: the model paper's/repo's fine-tuning recipe with a citation; else Recipe(lr=1e-4, weight_decay=0.01, epochs=100, batch_size=32) and recipe_source="harness default".
- `pretraining_corpora`: cite the model paper; state explicitly whether UCR/UEA data are included if the paper says so.
- Weights download to HF cache under TOR_CACHE (get_settings().cache_dir / "hf"; set HF_HOME accordingly before importing HF libs, or pass cache_dir). Call tor.config.load_secrets() before downloading (never print tokens).
- Install the model package with /data/hadar/envs/tor/bin/pip; if its pinned deps would downgrade torch/peft/transformers in a way that breaks other adapters, STOP and report NEEDS_CONTEXT instead of forcing it. Regenerate requirements.lock after installing.
- Contract tests run on CPU (small inputs). Do NOT use a GPU in this task.
- Add a model-specific test file tests/models/test_<family>.py (marker slow) checking: parameter count of the loaded pretrained model (print it in your report — it goes into the Phase 0 report), that `scratch` weights differ from pretrained weights for the same module, and the lora trainable-param count > head and < full.
- Commit per family: "Add <family> adapter". In the report list per adapter: total params, supported modes, lora_targets, grad ckpt support, max_length handling, recipe + source, pretraining corpora, any mode not supported and why.
