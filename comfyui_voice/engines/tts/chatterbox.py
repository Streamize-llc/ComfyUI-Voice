"""Chatterbox Multilingual v3 — emotion-controllable cloning TTS. MIT.

Subprocess-isolated: chatterbox-tts pins torch~2.6. Output is ALWAYS
Perth-watermarked (no off switch). Korean quality is acknowledged weaker by
Resemble — verify before production. Set up runtimes/chatterbox/.venv.
"""
from __future__ import annotations

from typing import Any

from ...base import BaseEngine, EngineCapabilities
from ...registry import register_engine


@register_engine("chatterbox")
class ChatterboxTTS(BaseEngine):
    CAPS = EngineCapabilities(
        id="chatterbox",
        display_name="Chatterbox Multilingual v3 (emotion, clone)",
        tasks=("tts",),
        license="MIT",
        commercial_safe=True,
        supports_cloning=True,
        needs_ref_audio=False,
        supports_emotion=True,
        languages=("ko", "en", "zh", "ja", "es", "fr", "de", "it", "pt", "ru"),
        sample_rate=24000,
        vram_est_gb=8.0,
        isolation="subprocess",
        max_input_chars=2000,
        dep_pins={"torch": "~=2.6"},
        pip_install=("chatterbox-tts", "git+https://github.com/resemble-ai/Perth.git@master"),
        probe_import=("chatterbox.mtl_tts",),
        param_schema={
            "params": {
                "text": {"type": "string", "multiline": True, "default": "안녕하세요, 만나서 반갑습니다.", "targets": ["text"]},
                "exaggeration": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "targets": ["exaggeration"]},
                "cfg_weight": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "targets": ["cfg_weight"]},
                "temperature": {"type": "float", "default": 0.8, "min": 0.1, "max": 1.5, "step": 0.05, "targets": ["temperature"]},
            }
        },
    )

    def load(self) -> None:
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
        self._sr = int(self._model.sr)

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        from ...audio_utils import write_temp_wav

        text = str(req.get("text") or "").strip()
        ref = req.get("ref_audio")
        kwargs: dict[str, Any] = {
            "language_id": req.get("language") if req.get("language") not in (None, "auto") else "ko",
            "exaggeration": float(req.get("exaggeration", 0.5)),
            "cfg_weight": float(req.get("cfg_weight", 0.5)),
            "temperature": float(req.get("temperature", 0.8)),
        }
        if ref:
            kwargs["audio_prompt_path"] = write_temp_wav(ref)
        wav = self._model.generate(text, **kwargs)
        return {"waveform": wav, "sample_rate": self._sr}
