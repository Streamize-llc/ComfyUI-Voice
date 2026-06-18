"""OuteTTS 1.0 (0.6B) — compact multilingual TTS w/ cloning. Apache-2.0 (0.6B).

Subprocess-isolated (outetts pins a specific stack). Korean is produced by
passing Hangul text; the default speaker is English-female. Set up
runtimes/oute_tts/.venv.
"""
from __future__ import annotations

from typing import Any

from ...base import BaseEngine, EngineCapabilities
from ...registry import register_engine


@register_engine("oute_tts")
class OuteTTS(BaseEngine):
    CAPS = EngineCapabilities(
        id="oute_tts",
        display_name="OuteTTS 1.0 (0.6B)",
        tasks=("tts",),
        license="Apache-2.0",
        commercial_safe=True,
        supports_cloning=False,
        languages=("ko", "en", "zh", "ja", "de", "es", "fr", "it"),
        sample_rate=44100,
        vram_est_gb=3.0,
        isolation="subprocess",
        max_input_chars=2000,
        pip_install=("outetts==0.4.4",),
        probe_import=("outetts",),
        param_schema={
            "params": {
                "text": {"type": "string", "multiline": True, "default": "안녕하세요, 만나서 반갑습니다.", "targets": ["text"]},
            }
        },
    )

    def load(self) -> None:
        import outetts

        self._outetts = outetts
        self._iface = outetts.Interface(
            config=outetts.ModelConfig.auto_config(
                model=outetts.Models.VERSION_1_0_SIZE_0_6B,
                backend=outetts.Backend.HF,
            )
        )
        self._speaker = self._iface.load_default_speaker("EN-FEMALE-1-NEUTRAL")

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        o = self._outetts
        text = str(req.get("text") or "").strip()
        cfg = o.GenerationConfig(
            text=text,
            generation_type=o.GenerationType.CHUNKED,
            speaker=self._speaker,
        )
        out = self._iface.generate(config=cfg)
        audio = out.audio
        arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
        return {"waveform": arr.astype(np.float32).reshape(-1), "sample_rate": int(getattr(out, "sr", 44100))}
