"""Engine adapter registry — the core extensibility mechanism.

Adding a model is a single-file drop-in: put ``<engine>.py`` under
``engines/<task>/``, declare ``CAPS`` and decorate the class with
``@register_engine("<id>")``. ``scan_engines()`` folder-scans and imports them,
guarding each import so one broken adapter never takes down the others.

Mirrors this repo's ``comfyui_api/workflows.py`` registry idea: a declaration
drives the form/API/MCP uniformly, with no central if/elif dispatch.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Callable

from .base import BaseEngine

log = logging.getLogger("comfyui_voice.registry")

ENGINE_REGISTRY: dict[str, type[BaseEngine]] = {}
TASK_INDEX: dict[str, list[str]] = {}  # "tts" -> ["reference_tone", ...]

_scanned = False


def register_engine(engine_id: str) -> Callable[[type[BaseEngine]], type[BaseEngine]]:
    """Class decorator: self-register an engine adapter under ``engine_id``."""

    def deco(cls: type[BaseEngine]) -> type[BaseEngine]:
        caps = getattr(cls, "CAPS", None)
        if caps is None:
            raise ValueError(f"{cls.__name__} must declare a CAPS = EngineCapabilities(...)")
        if caps.id != engine_id:
            raise ValueError(
                f"{cls.__name__}: CAPS.id ({caps.id!r}) must equal "
                f"@register_engine id ({engine_id!r})"
            )
        existing = ENGINE_REGISTRY.get(engine_id)
        if existing is not None and existing is not cls:
            raise ValueError(f"engine id collision: {engine_id!r} already registered")
        caps.validate()
        ENGINE_REGISTRY[engine_id] = cls
        for task in caps.tasks:
            bucket = TASK_INDEX.setdefault(task, [])
            if engine_id not in bucket:
                bucket.append(engine_id)
        log.debug("registered engine %s (tasks=%s)", engine_id, caps.tasks)
        return cls

    return deco


def scan_engines(force: bool = False) -> None:
    """Import every module under ``comfyui_voice.engines`` so adapters register.

    Import failures are logged and skipped (a missing model dependency must not
    break the whole suite).
    """
    global _scanned
    if _scanned and not force:
        return
    from . import engines as engines_pkg

    for modinfo in pkgutil.walk_packages(engines_pkg.__path__, engines_pkg.__name__ + "."):
        if modinfo.ispkg:
            continue
        try:
            importlib.import_module(modinfo.name)
        except Exception as exc:  # noqa: BLE001 - intentional per-adapter guard
            log.warning("skipping engine module %s: %s", modinfo.name, exc)
    _scanned = True


def get_engine(engine_id: str) -> type[BaseEngine]:
    if engine_id not in ENGINE_REGISTRY:
        raise KeyError(
            f"unknown engine {engine_id!r}; installed: {sorted(ENGINE_REGISTRY)}"
        )
    return ENGINE_REGISTRY[engine_id]


def engines_for_task(task: str) -> list[str]:
    return list(TASK_INDEX.get(task, []))


def all_capabilities() -> dict:
    return {eid: cls.CAPS for eid, cls in ENGINE_REGISTRY.items()}


def engine_available(engine_id: str) -> bool:
    """Whether an engine can actually run now (deps present / venv set up).

    Adapters always REGISTER (CAPS is dep-free); this reports whether the heavy
    deps are installed so the UI can show 'available' vs 'pip install ...'.
    """
    import importlib.util

    caps = get_engine(engine_id).CAPS
    if caps.isolation == "subprocess":
        from .paths import venv_python

        return venv_python(engine_id) is not None
    if not caps.probe_import:
        return True
    for module in caps.probe_import:
        try:
            if importlib.util.find_spec(module) is None:
                return False
        except (ImportError, ValueError, ModuleNotFoundError):
            return False
    return True


def install_hint(engine_id: str) -> str:
    """A human-actionable hint for enabling an unavailable engine."""
    caps = get_engine(engine_id).CAPS
    if caps.isolation == "subprocess":
        return (
            f"subprocess engine — set up its venv under runtimes/{engine_id}/.venv "
            f"and install: {' '.join(caps.pip_install) or '(see adapter)'}"
        )
    if caps.pip_install:
        return "pip install " + " ".join(caps.pip_install)
    return "(no install metadata)"
