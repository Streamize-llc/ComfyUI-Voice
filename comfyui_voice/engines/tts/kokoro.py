"""Kokoro-82M — ultralight multilingual TTS. Apache-2.0.

NOTE: Kokoro does NOT support Korean (no 'ko' lang_code / no Korean voice). It is
included for English/JA/ZH/EU presets. For Korean use melotts_korean / cosyvoice3
/ qwen3_tts. In-process (no hard pins) — but run `pip install --dry-run` first.
"""
from __future__ import annotations

from typing import Any

from ...base import BaseEngine, EngineCapabilities
from ...registry import register_engine

_LANG = {"en": "a", "es": "e", "fr": "f", "hi": "h", "it": "i", "pt": "p", "ja": "j", "zh": "z"}
_VOICE = {"a": "af_heart", "e": "ef_dora", "f": "ff_siwis", "h": "hf_alpha",
          "i": "if_sara", "p": "pf_dora", "j": "jf_alpha", "z": "zf_xiaobei"}


@register_engine("kokoro")
class KokoroTTS(BaseEngine):
    CAPS = EngineCapabilities(
        id="kokoro",
        display_name="Kokoro-82M (no Korean)",
        tasks=("tts",),
        license="Apache-2.0",
        commercial_safe=True,
        supports_cloning=False,
        languages=tuple(_LANG.keys()),  # deliberately NO 'ko'
        sample_rate=24000,
        vram_est_gb=1.0,
        isolation="inproc",
        max_input_chars=2000,
        pip_install=("kokoro>=0.9.4", "soundfile", "misaki[en]>=0.9.4"),
        probe_import=("kokoro",),
        param_schema={
            "params": {
                "text": {"type": "string", "multiline": True, "default": "Hello from Kokoro.", "targets": ["text"]},
                "voice": {"type": "string", "default": "af_heart", "targets": ["voice"]},
                "speed": {"type": "float", "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "targets": ["speed"]},
            }
        },
    )

    def load(self) -> None:
        from kokoro import KPipeline

        self._KPipeline = KPipeline
        self._pipes: dict[str, Any] = {}

    def _pipe(self, lang_code: str):
        if lang_code not in self._pipes:
            self._pipes[lang_code] = self._KPipeline(lang_code=lang_code)
        return self._pipes[lang_code]

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        import torch

        text = str(req.get("text") or "").strip()
        lang = req.get("language") or "en"
        if lang in (None, "auto"):
            lang = "en"
        lang_code = _LANG.get(lang)
        if lang_code is None:
            raise ValueError(
                f"Kokoro does not support language '{lang}' (no Korean). "
                f"Supported: {sorted(_LANG)}"
            )
        voice = req.get("voice") or _VOICE[lang_code]
        speed = float(req.get("speed", 1.0)) or 1.0
        chunks = [audio for _, _, audio in self._pipe(lang_code)(text, voice=voice, speed=speed)]
        wav = torch.cat(chunks) if chunks else torch.zeros(1)
        return {"waveform": wav, "sample_rate": 24000}
