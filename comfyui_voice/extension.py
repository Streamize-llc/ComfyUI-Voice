"""V3 ComfyExtension entrypoint for the ComfyUI-Voice suite."""
from __future__ import annotations

import logging

from comfy_api.latest import IO, ComfyExtension

from .registry import scan_engines

log = logging.getLogger("comfyui_voice")


class VoiceExtension(ComfyExtension):
    async def on_load(self) -> None:
        scan_engines()
        from .registry import ENGINE_REGISTRY

        names = ", ".join(sorted(ENGINE_REGISTRY)) or "none"
        log.info("ComfyUI-Voice: %d engine(s) registered: %s", len(ENGINE_REGISTRY), names)

    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        from .nodes.asr import VoiceASR
        from .nodes.tts import VoiceTTS
        from .nodes.util import VoiceEngineInfo

        return [VoiceTTS, VoiceASR, VoiceEngineInfo]


async def comfy_entrypoint() -> VoiceExtension:
    return VoiceExtension()
