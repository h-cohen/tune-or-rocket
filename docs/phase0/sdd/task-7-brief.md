### Task 7: Adapter interface, training loop, InceptionTime reference

**Files:** Create `tor/models/__init__.py` (registry), `tor/models/base.py`, `tor/models/inceptiontime.py`, `tor/train.py`, `tests/test_train.py`, `tests/models/test_contract.py`

**Interfaces:**
- Produces:
  - `Mode = Literal["head","lora","full","scratch"]`.
  - `class FMAdapter(ABC)`: class attrs `name: str`, `hf_repo: str | None`, `max_length: int | None`, `multivariate: Literal["native","per_channel"]`, `pretraining_corpora: str` (cited), `recipe_source: str`; methods `load(init: Literal["pretrained"] | tuple[Literal["random"], int]) -> None`, `preprocess(X: np.ndarray) -> torch.Tensor`, `embed(X: torch.Tensor) -> torch.Tensor` (shape `(n, d)`), `classifier(n_classes: int, mode: Mode, grad_ckpt: bool = False) -> torch.nn.Module` (the `scratch` mode loads `("random", seed)` first, then trains like `full`), `default_recipe() -> Recipe`, `lora_config() -> peft.LoraConfig` (default `r=8, lora_alpha=16, target_modules` = the model's attention q/k/v/o projection names, dropout 0.0).
  - `Recipe(lr: float, weight_decay: float, epochs: int, batch_size: int, warmup_frac: float = 0.0)`.
  - `REGISTRY: dict[str, type[FMAdapter]]`; `register(cls)` decorator; `available_adapters() -> list[str]` (skips any adapter whose import or availability check fails, logging the reason).
  - `count_params(module) -> tuple[int, int]` (total, trainable).
  - `train_epochs(model, X: torch.Tensor, y: torch.Tensor, recipe: Recipe, epochs: int, device: str, amp: bool = True) -> TrainResult(losses: list[float], epoch_times_s: list[float], diverged: bool)`, with AdamW, cross-entropy and bf16 autocast if `amp`. `diverged` is True on a non-finite loss, and training stops at that point. No evaluation function exists in this module.
  - `InceptionTime` adapter (`name="inceptiontime"`, in PyTorch: 6 inception modules of depth 3, 32 filters, kernel sizes 40/20/10, residual every 3, GAP head). Here `pretrained` is not supported and only the `scratch` mode exists.
- [ ] **Step 1: Write failing tests** in `tests/test_train.py`, using a tiny `ToyAdapter` defined in the test (a linear encoder):
  - `test_modes_set_trainable_params`: in `head` mode only the head params train; in `full` mode they all do; `count_params` reflects this.
  - `test_train_epochs_records_times_and_losses`: 3 epochs on random data give `len(losses) == 3`, `len(epoch_times_s) == 3` and `diverged is False`.
  - `test_divergence_detected`: lr=1e6 on toy data gives `diverged is True`, with fewer than the requested epochs recorded.
  - `test_no_eval_api`: `not hasattr(tor.train, "evaluate")` and `not hasattr(tor.train, "accuracy")`.
- [ ] **Step 2: Write** `tests/models/test_contract.py`, parametrized over `available_adapters()` and marked `slow`. For each adapter and each mode it supports: `classifier(3, mode)` on `preprocess(randn(4,1,128))` returns logits of shape `(4,3)`; the trainable fraction matches the mode (`head` < `lora` < `full`); `embed` returns 2-D. InceptionTime must pass now.
- [ ] **Step 3: Run** `pytest tests/test_train.py tests/models -v`. Expected: FAIL.
- [ ] **Step 4: Implement** the base class, registry, loop and InceptionTime.
- [ ] **Step 5: Run** `pytest tests/test_train.py tests/models -v -m "slow or not slow"`. Expected: all passed.
- [ ] **Step 6: Commit** `"Add adapter interface, training loop and InceptionTime reference"`.

