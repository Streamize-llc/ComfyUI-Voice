"""Reference TTS engine — dependency-free, deterministic.

This is NOT real speech. It maps text to a deterministic sequence of tones using
nothing but torch. It exists to:

1. Prove the whole pipeline end-to-end with ZERO model dependencies (so v0 is
   verifiable on a clean install, and the pure-core test has something to run).
2. Be the canonical template a contributor copies to add a real engine —
   the structure (CAPS declaration + load/generate/unload) is exactly what a
   real adapter looks like. Replace the body of ``generate`` with a model call.

To add a real engine: copy this file to ``engines/tts/<name>.py``, lazy-import
the model inside ``load()``, fill ``generate()``, and update ``CAPS``.
"""
from __future__ import annotations

import math
from typing import Any

import torch

from ...base import BaseEngine, EngineCapabilities
from ...registry import register_engine

# Korean + latin vowels get a longer note, so output has speech-like rhythm.
_VOWELS = set("aeiouAEIOU아야어여오요우유으이애에왜외워")


@register_engine("reference_tone")
class ReferenceToneTTS(BaseEngine):
    CAPS = EngineCapabilities(
        id="reference_tone",
        display_name="Reference Tone (demo · no deps)",
        tasks=("tts",),
        version="1",
        license="Apache-2.0",
        commercial_safe=True,
        supports_cloning=False,
        languages=("any",),
        sample_rate=24000,
        output_channels=1,
        vram_est_gb=0.0,
        isolation="inproc",
        max_input_chars=2000,
        param_schema={
            "params": {
                "text": {
                    "type": "string",
                    "multiline": True,
                    "default": "안녕하세요. ComfyUI Voice 입니다.",
                    "targets": ["text"],
                },
                "seed": {"type": "int", "default": 42, "min": 0, "max": 2**32 - 1, "targets": ["seed"]},
                "speed": {
                    "type": "float",
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.05,
                    "targets": ["speed"],
                },
            }
        },
    )

    def load(self) -> None:
        self._loaded = True

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        text = str(req.get("text") or "").strip() or "..."
        seed = int(req.get("seed", 42))
        speed = float(req.get("speed", 1.0)) or 1.0
        sr = self.CAPS.sample_rate
        base_freq = 196.0  # G3

        segments: list[torch.Tensor] = []
        fade = max(1, int(0.01 * sr))
        for i, ch in enumerate(text):
            if ch.isspace():
                segments.append(torch.zeros(int(sr * 0.06 / speed)))
                continue
            semitone = (ord(ch) + seed * 7 + i) % 24
            freq = base_freq * (2.0 ** (semitone / 12.0))
            dur = 0.13 if ch in _VOWELS else 0.085
            n = max(fade * 2 + 1, int(sr * dur / speed))
            t = torch.arange(n, dtype=torch.float32) / sr
            wave = 0.3 * torch.sin(2 * math.pi * freq * t)
            wave += 0.1 * torch.sin(2 * math.pi * 2 * freq * t)  # soft 2nd harmonic
            env = torch.ones(n)
            env[:fade] = torch.linspace(0.0, 1.0, fade)
            env[-fade:] = torch.linspace(1.0, 0.0, fade)
            segments.append(wave * env)

        waveform = torch.cat(segments) if segments else torch.zeros(int(sr * 0.2))
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak * 0.9
        return {"waveform": waveform, "sample_rate": sr}
