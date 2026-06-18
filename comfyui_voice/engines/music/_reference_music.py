"""Reference music engine — dependency-free, deterministic.

NOT real music. Synthesizes a deterministic chord loop from (seed, duration) with
nothing but torch, so the music-generation pipeline (VoiceMusicGen node + run_gen)
is verifiable on a clean install and serves as the adapter template for a real
engine (e.g. ACE-Step). Replace generate() with a model call.
"""
from __future__ import annotations

import math
from typing import Any

from ...base import BaseEngine, EngineCapabilities
from ...registry import register_engine


@register_engine("reference_music")
class ReferenceMusic(BaseEngine):
    CAPS = EngineCapabilities(
        id="reference_music",
        display_name="Reference Music (demo · no deps)",
        tasks=("music",),
        version="1",
        license="Apache-2.0",
        commercial_safe=True,
        languages=("any",),
        sample_rate=44100,
        vram_est_gb=0.0,
        isolation="inproc",
        param_schema={
            "params": {
                "text": {"type": "string", "multiline": True, "default": "calm lofi piano, 90 bpm", "targets": ["text"]},
                "duration": {"type": "float", "default": 8.0, "min": 1.0, "max": 240.0, "step": 1.0, "targets": ["duration"]},
                "seed": {"type": "int", "default": 42, "min": 0, "max": 2**32 - 1, "targets": ["seed"]},
            }
        },
    )

    def load(self) -> None:
        self._loaded = True

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        import torch

        dur = max(0.5, float(req.get("duration", 8.0)))
        seed = int(req.get("seed", 42))
        sr = self.CAPS.sample_rate
        n = int(sr * dur)
        t = torch.arange(n, dtype=torch.float32) / sr
        root = 220.0 * (2.0 ** ((seed % 12) / 12.0))  # seed picks the key
        chord = sum(0.2 * torch.sin(2 * math.pi * f * t) for f in (root, root * 5 / 4, root * 3 / 2))
        chord = chord * (0.6 + 0.4 * torch.sin(2 * math.pi * 2.0 * t))  # tremolo
        peak = chord.abs().max()
        if peak > 0:
            chord = chord / peak * 0.7
        return {"waveform": chord, "sample_rate": sr}
