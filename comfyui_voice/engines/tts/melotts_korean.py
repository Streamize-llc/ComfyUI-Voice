"""MeloTTS (Korean) — fast, lightweight preset Korean TTS. MIT.

Subprocess-isolated: MeloTTS hard-pins transformers==4.27.4 + librosa==0.9.1
(numpy<2), which conflicts with the host stack. Set up its venv at
runtimes/melotts_korean/.venv and install the pip_install packages there.
"""
from __future__ import annotations

from typing import Any

from ...base import BaseEngine, EngineCapabilities
from ...registry import register_engine


@register_engine("melotts_korean")
class MeloTTSKorean(BaseEngine):
    CAPS = EngineCapabilities(
        id="melotts_korean",
        display_name="MeloTTS (Korean)",
        tasks=("tts",),
        license="MIT",
        commercial_safe=True,
        supports_cloning=False,
        languages=("ko",),
        sample_rate=44100,
        vram_est_gb=1.5,
        isolation="subprocess",
        max_input_chars=2000,
        dep_pins={"transformers": "==4.27.4", "numpy": "<2"},
        pip_install=(
            "git+https://github.com/myshell-ai/MeloTTS.git",
            "python-mecab-ko",
            "python-mecab-ko-dic",
        ),
        probe_import=("melo",),
        param_schema={
            "params": {
                "text": {"type": "string", "multiline": True, "default": "안녕하세요. 멜로 TTS 입니다.", "targets": ["text"]},
                "speed": {"type": "float", "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "targets": ["speed"]},
            }
        },
    )

    def load(self) -> None:
        from melo.api import TTS

        self._tts = TTS(language="KR", device="auto")
        self._spk = self._tts.hps.data.spk2id["KR"]
        self._sr = int(self._tts.hps.data.sampling_rate)

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        text = str(req.get("text") or "").strip()
        speed = float(req.get("speed", 1.0)) or 1.0
        wav = self._tts.tts_to_file(text, self._spk, output_path=None, speed=speed, quiet=True)
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        return {"waveform": wav, "sample_rate": self._sr}
