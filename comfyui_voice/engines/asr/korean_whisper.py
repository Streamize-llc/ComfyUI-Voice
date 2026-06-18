"""Korean Whisper fine-tune (ghost613) via faster-whisper — best-Korean ASR. Apache-2.0.

In-process (CTranslate2). Subclasses the faster-whisper adapter, swapping the
model id and defaulting the language to Korean — the GGUF-style "base node
subclass" extension pattern.
"""
from __future__ import annotations

from typing import Any

from ...base import EngineCapabilities
from ...registry import register_engine
from .faster_whisper import FasterWhisper


@register_engine("korean_whisper")
class KoreanWhisper(FasterWhisper):
    MODEL_ID = "ghost613/faster-whisper-large-v3-turbo-korean"

    CAPS = EngineCapabilities(
        id="korean_whisper",
        display_name="Korean Whisper (ghost613, faster-whisper)",
        tasks=("asr",),
        license="Apache-2.0",
        commercial_safe=True,
        languages=("ko",),
        sample_rate=16000,
        vram_est_gb=6.0,
        isolation="inproc",
        pip_install=("faster-whisper>=1.1.0",),
        probe_import=("faster_whisper",),
        param_schema={"params": {}},
    )

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        req = dict(req)
        if req.get("language") in (None, "auto"):
            req["language"] = "ko"
        return super().generate(task, req)
