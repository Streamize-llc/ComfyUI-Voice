"""Generative-audio orchestration: text (+ duration/seed) -> AUDIO.

Shared by the music and SFX generation nodes. Adapters only implement a raw
generate(); chunking/long-form for very long music can land here later.
"""
from __future__ import annotations

import logging

from ..audio_utils import ensure_sr, to_audio_dict
from ..registry import get_engine
from ..runtime import get_runtime

log = logging.getLogger("comfyui_voice.gen")


def run_gen(
    task: str,
    engine_id: str,
    *,
    text: str,
    duration: float = 10.0,
    seed: int = 42,
    params: dict | None = None,
    target_sr: int | None = None,
) -> dict:
    """Generate audio from a text prompt with ``engine_id`` (task = music|sfx)."""
    caps = get_engine(engine_id).CAPS
    req = {"text": text, "duration": float(duration), "seed": int(seed), **(params or {})}
    out = get_runtime(engine_id).generate(task, req)
    audio = to_audio_dict(out["waveform"], out.get("sample_rate", caps.sample_rate))
    if target_sr:
        audio = ensure_sr(audio, target_sr)
    return audio
