"""Adapter registry."""
from __future__ import annotations

import importlib
import logging

from tor.models.base import FMAdapter, Mode, Recipe

log = logging.getLogger(__name__)

REGISTRY: dict[str, type[FMAdapter]] = {}

# Adapter modules; each registers itself on import. Imports may fail if an
# optional dependency is missing; available_adapters() then skips with a warning.
_ADAPTER_MODULES = ["tor.models.inceptiontime"]


def register(cls: type[FMAdapter]) -> type[FMAdapter]:
    REGISTRY[cls.name] = cls
    return cls


def available_adapters() -> list[str]:
    for mod in _ADAPTER_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as e:  # missing optional dependency etc.
            log.warning("adapter module %s unavailable: %s", mod, e)
    return sorted(REGISTRY)


__all__ = ["FMAdapter", "Mode", "Recipe", "REGISTRY", "register", "available_adapters"]
