"""Adapter interface for foundation-model (FM) classifiers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

Mode = Literal["head", "lora", "full", "scratch"]
ALL_MODES: tuple[Mode, ...] = ("head", "lora", "full", "scratch")


@dataclass(frozen=True)
class Recipe:
    lr: float
    weight_decay: float
    epochs: int
    batch_size: int
    warmup_frac: float = 0.0


class FMAdapter(ABC):
    name: str
    hf_repo: str | None
    max_length: int | None
    multivariate: Literal["native", "per_channel"]
    pretraining_corpora: str  # cited
    recipe_source: str
    supported_modes: tuple[Mode, ...] = ALL_MODES
    # Names of attention q/k/v/o projection modules, used for LoRA.
    lora_targets: tuple[str, ...] = ()

    @abstractmethod
    def load(self, init: Literal["pretrained"] | tuple[Literal["random"], int]) -> None: ...

    @abstractmethod
    def preprocess(self, X: np.ndarray) -> torch.Tensor: ...

    @abstractmethod
    def embed(self, X: torch.Tensor) -> torch.Tensor:
        """Return features of shape (n, d)."""

    @abstractmethod
    def classifier(
        self, n_classes: int, mode: Mode, grad_ckpt: bool = False, seed: int = 0
    ) -> torch.nn.Module:
        """Build a classifier. `scratch` loads ("random", seed) first, then trains like `full`.

        Implementations must call `self._check_mode(mode)` first.
        """

    @abstractmethod
    def default_recipe(self) -> Recipe: ...

    def _check_mode(self, mode: str) -> None:
        if mode not in self.supported_modes:
            raise ValueError(
                f"{self.name} does not support mode {mode!r}; supported: {self.supported_modes}"
            )

    def lora_config(self):
        """LoRA config (r=8, alpha=16, dropout 0). Only meaningful if "lora" is supported."""
        if not self.lora_targets:
            raise NotImplementedError(f"{self.name} defines no lora_targets")
        import peft

        return peft.LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, target_modules=list(self.lora_targets)
        )
