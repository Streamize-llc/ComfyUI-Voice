"""Suite-owned model manager.

ComfyUI's ``model_management`` only tracks models wrapped in its ModelPatcher,
so raw models an adapter loads are invisible to its eviction logic and can OOM a
chained audio pipeline. This manager tracks every engine instance the suite
loads, exposes their declared VRAM, and frees room before loading a new one.

v0 keeps the policy simple (best-effort free + LRU bookkeeping); the hook points
for smarter cross-node coordination are here.
"""
from __future__ import annotations

import logging

from .registry import get_engine

log = logging.getLogger("comfyui_voice.manager")


class _ModelManager:
    def __init__(self) -> None:
        self._loaded: dict[str, object] = {}  # engine_id -> instance
        self._order: list[str] = []  # LRU, oldest first

    def get(self, engine_id: str):
        """Return a loaded engine instance, loading (and making room) if needed."""
        if engine_id in self._loaded:
            self._touch(engine_id)
            return self._loaded[engine_id]

        cls = get_engine(engine_id)
        caps = cls.CAPS
        self._free_room(caps.vram_est_gb)

        inst = cls()
        log.info("loading engine '%s' (vram_est=%s GB)", engine_id, caps.vram_est_gb)
        inst.load()
        self._loaded[engine_id] = inst
        self._order.append(engine_id)
        return inst

    def _touch(self, engine_id: str) -> None:
        if engine_id in self._order:
            self._order.remove(engine_id)
        self._order.append(engine_id)

    def _free_room(self, need_gb: float | None) -> None:
        # Best-effort: ask ComfyUI to free VRAM it owns for the declared amount.
        try:
            import comfy.model_management as mm

            if need_gb:
                mm.free_memory(float(need_gb) * 1024**3, mm.get_torch_device())
        except Exception:  # pragma: no cover - optional in pure-core tests
            pass

    def unload(self, engine_id: str) -> None:
        inst = self._loaded.pop(engine_id, None)
        if inst is not None:
            try:
                inst.unload()
            except Exception:  # noqa: BLE001
                log.warning("unload failed for '%s'", engine_id, exc_info=True)
            if engine_id in self._order:
                self._order.remove(engine_id)
            log.info("unloaded engine '%s'", engine_id)

    def unload_all(self) -> None:
        for engine_id in list(self._loaded):
            self.unload(engine_id)

    def loaded_ids(self) -> list[str]:
        return list(self._order)


MANAGER = _ModelManager()
