"""Adapter registry."""
from __future__ import annotations

import importlib
import logging

from tor.models.base import AdapterUnavailable, FMAdapter, Mode, Recipe

log = logging.getLogger(__name__)

REGISTRY: dict[str, type[FMAdapter]] = {}

# Adapter modules; each registers itself on import. Imports may fail if an
# optional dependency is missing; available_adapters() then skips with a warning.
_ADAPTER_MODULES = ["tor.models.inceptiontime"]


def register(cls: type[FMAdapter]) -> type[FMAdapter]:
    if cls.name in REGISTRY and REGISTRY[cls.name] is not cls:
        raise ValueError(f"duplicate adapter name {cls.name!r}")
    REGISTRY[cls.name] = cls
    return cls


def _import_all() -> None:
    for mod in _ADAPTER_MODULES:
        try:
            importlib.import_module(mod)
        except (ImportError, AdapterUnavailable) as e:
            log.warning("adapter module %s unavailable: %s", mod, e)


def get_adapter(name: str) -> type[FMAdapter]:
    _import_all()
    return REGISTRY[name]


def available_adapters() -> list[str]:
    """Names of usable adapters. Only ImportError/AdapterUnavailable/is_available()=False skip."""
    _import_all()
    out = []
    for name in sorted(REGISTRY):
        ok, reason = REGISTRY[name].is_available()
        if ok:
            out.append(name)
        else:
            log.warning("adapter %s unavailable: %s", name, reason)
    return out


__all__ = ["AdapterUnavailable", "FMAdapter", "Mode", "get_adapter", "Recipe", "REGISTRY", "register", "available_adapters"]
