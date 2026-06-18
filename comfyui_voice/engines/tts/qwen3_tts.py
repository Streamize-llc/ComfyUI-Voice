"""Qwen3-TTS (1.7B CustomVoice) — flagship, native Korean preset voices. Apache-2.0.

Subprocess-isolated: the qwen-tts stack/codec deps can move transformers; keep it
in runtimes/qwen3_tts/.venv. CustomVoice = preset speakers (e.g. "Sohee" = native
Korean). Cloning/voice-design are separate checkpoints (future adapters).
"""
from __future__ import annotations

from typing import Any

from ...base import BaseEngine, EngineCapabilities
from ...registry import register_engine

_SPEAKERS = ["Sohee", "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna"]


@register_engine("qwen3_tts")
class Qwen3TTS(BaseEngine):
    CAPS = EngineCapabilities(
        id="qwen3_tts",
        display_name="Qwen3-TTS 1.7B (CustomVoice)",
        tasks=("tts",),
        license="Apache-2.0",
        commercial_safe=True,
        supports_cloning=False,  # CustomVoice = preset speakers; clone is a separate ckpt
        supports_emotion=True,
        languages=("ko", "zh", "en", "ja", "de", "es", "fr", "it", "ru", "pt"),
        sample_rate=24000,
        vram_est_gb=6.0,
        isolation="subprocess",
        max_input_chars=2000,
        pip_install=("qwen-tts", "soundfile"),
        probe_import=("qwen_tts",),
        param_schema={
            "params": {
                "text": {"type": "string", "multiline": True, "default": "안녕하세요, 만나서 반갑습니다.", "targets": ["text"]},
                "speaker": {"type": "enum", "default": "Sohee", "options": _SPEAKERS, "targets": ["speaker"]},
                "instruct": {"type": "string", "default": "", "targets": ["instruct"]},
            }
        },
    )

    def load(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        repo = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        try:
            self._model = Qwen3TTSModel.from_pretrained(
                repo, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2"
            )
        except Exception:
            self._model = Qwen3TTSModel.from_pretrained(
                repo, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
            )

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        text = str(req.get("text") or "").strip()
        speaker = req.get("speaker") or "Sohee"
        instruct = str(req.get("instruct") or "")
        wavs, sr = self._model.generate_custom_voice(
            text=text, language="Korean", speaker=speaker, instruct=instruct
        )
        return {"waveform": np.asarray(wavs[0], dtype=np.float32).reshape(-1), "sample_rate": int(sr)}
