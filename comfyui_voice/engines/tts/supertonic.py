"""Supertonic 3 — fast on-device TTS, strong native Korean (Supertone/HYBE).

In-process (onnxruntime, no host-stack conflict). License: code MIT, model
OpenRAIL-M (use restrictions) — review before commercial use. Voices are
language-agnostic style vectors (M1-M5 / F1-F5).
"""
from __future__ import annotations

from typing import Any

from ...base import BaseEngine, EngineCapabilities
from ...registry import register_engine

_VOICES = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]


@register_engine("supertonic")
class SupertonicTTS(BaseEngine):
    CAPS = EngineCapabilities(
        id="supertonic",
        display_name="Supertonic 3 (on-device, KO)",
        tasks=("tts",),
        license="OpenRAIL-M (model) / MIT (code)",
        commercial_safe=False,  # OpenRAIL-M use restrictions — review
        supports_cloning=False,
        languages=("ko", "en", "ja", "zh", "es", "fr", "de"),
        sample_rate=44100,
        vram_est_gb=0.5,
        isolation="inproc",
        max_input_chars=2000,
        pip_install=("supertonic",),
        probe_import=("supertonic",),
        param_schema={
            "params": {
                "text": {"type": "string", "multiline": True, "default": "안녕하세요, 만나서 반갑습니다.", "targets": ["text"]},
                "voice": {"type": "enum", "default": "F1", "options": _VOICES, "targets": ["voice"]},
                "speed": {"type": "float", "default": 1.05, "min": 0.7, "max": 2.0, "step": 0.05, "targets": ["speed"]},
            }
        },
    )

    def load(self) -> None:
        from supertonic import TTS

        self._tts = TTS(model="supertonic-3", auto_download=True)

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        text = str(req.get("text") or "").strip()
        voice = req.get("voice") or "F1"
        lang = req.get("language")
        lang = "ko" if lang in (None, "auto") else lang
        style = self._tts.get_voice_style(voice_name=voice)
        wav, _ = self._tts.synthesize(
            text=text, voice_style=style, total_steps=8,
            speed=float(req.get("speed", 1.05)), silence_duration=0.3, lang=lang, verbose=False,
        )
        return {"waveform": np.asarray(wav, dtype=np.float32), "sample_rate": int(self._tts.sample_rate)}
