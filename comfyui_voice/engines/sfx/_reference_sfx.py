"""Reference SFX engine — dependency-free, deterministic.

NOT real sound design. Produces a deterministic enveloped-noise burst from
(seed, duration) so the SFX pipeline (VoiceSFXGen node + run_gen) is verifiable
on a clean install and serves as the adapter template for a real engine (e.g.
MOSS-SoundEffect). Replace generate() with a model call.
"""
from __future__ import annotations

from typing import Any

from ...base import BaseEngine, EngineCapabilities
from ...registry import register_engine


@register_engine("reference_sfx")
class ReferenceSFX(BaseEngine):
    CAPS = EngineCapabilities(
        id="reference_sfx",
        display_name="Reference SFX (demo · no deps)",
        tasks=("sfx",),
        version="1",
        license="Apache-2.0",
        commercial_safe=True,
        languages=("any",),
        sample_rate=44100,
        vram_est_gb=0.0,
        isolation="inproc",
        param_schema={
            "params": {
                "text": {"type": "string", "multiline": True, "default": "rain on a window, distant thunder", "targets": ["text"]},
                "duration": {"type": "float", "default": 3.0, "min": 0.5, "max": 60.0, "step": 0.5, "targets": ["duration"]},
                "seed": {"type": "int", "default": 42, "min": 0, "max": 2**32 - 1, "targets": ["seed"]},
            }
        },
    )

    def load(self) -> None:
        self._loaded = True

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        import torch

        dur = max(0.25, float(req.get("duration", 3.0)))
        seed = int(req.get("seed", 42))
        sr = self.CAPS.sample_rate
        n = int(sr * dur)
        gen = torch.Generator().manual_seed(seed)
        noise = torch.rand(n, generator=gen) * 2.0 - 1.0
        env = torch.exp(-torch.arange(n, dtype=torch.float32) / (sr * dur * 0.3))  # decay
        wav = noise * env * 0.7
        return {"waveform": wav, "sample_rate": sr}
