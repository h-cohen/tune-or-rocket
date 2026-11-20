"""Adapter interface for foundation-model (FM) classifiers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

Mode = Literal["head", "lora", "full", "scratch"]
ALL_MODES: tuple[Mode, ...] = ("head", "lora", "full", "scratch")


class AdapterUnavailable(Exception):
    """Raised on import (or reported by is_available) when an adapter cannot be used."""


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
    supports_grad_ckpt: bool = False
    # Names of attention q/k/v/o projection modules, used for LoRA.
    lora_targets: tuple[str, ...] = ()

    @abstractmethod
    def load(self, init: Literal["pretrained"] | tuple[Literal["random"], int]) -> None: ...

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        """(available, reason-if-not). Default: always available."""
        return True, ""

    @abstractmethod
    def preprocess(self, X: np.ndarray) -> torch.Tensor:
        """Return a float32 tensor of shape (n, c, L').

        How series longer than `max_length` are handled is adapter-specific and
        documented per adapter.
        """

    @abstractmethod
    def embed(self, X: torch.Tensor) -> torch.Tensor:
        """Return features of shape (n, d).

        Per-channel adapters concatenate channel embeddings (d = d_model * C)
        unless the adapter documents otherwise.
        """

    @abstractmethod
    def classifier(
        self,
        n_classes: int,
        mode: Mode,
        n_channels: int,
        seq_len: int,
        grad_ckpt: bool = False,
        seed: int = 0,
    ) -> torch.nn.Module:
        """Build a classifier for inputs of shape (n, n_channels, seq_len).

        `seq_len` is the length AFTER preprocess. `scratch` loads ("random", seed) first, then trains like `full`.

        Implementations must call `self._check_mode(mode, grad_ckpt)` first.
        """

    @abstractmethod
    def default_recipe(self) -> Recipe: ...

    def _check_mode(self, mode: str, grad_ckpt: bool = False) -> None:
        if grad_ckpt and not self.supports_grad_ckpt:
            raise ValueError(f"{self.name} does not support gradient checkpointing")
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
