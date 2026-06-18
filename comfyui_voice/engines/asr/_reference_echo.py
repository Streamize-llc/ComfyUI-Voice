"""Reference ASR engine — dependency-free, deterministic.

Does NOT actually transcribe. It reports the clip's duration as a placeholder
transcript so the VoiceASR node + VOICE_TRANSCRIPT type + asr_processor are
verifiable end-to-end with zero model dependencies, and so contributors have a
template to copy for a real ASR adapter (e.g. faster-whisper).
"""
from __future__ import annotations

from typing import Any

from ...base import BaseEngine, EngineCapabilities
from ...registry import register_engine


@register_engine("reference_asr")
class ReferenceEchoASR(BaseEngine):
    CAPS = EngineCapabilities(
        id="reference_asr",
        display_name="Reference ASR (demo · no deps)",
        tasks=("asr",),
        version="1",
        license="Apache-2.0",
        commercial_safe=True,
        languages=("any",),
        sample_rate=16000,
        isolation="inproc",
    )

    def load(self) -> None:
        self._loaded = True

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        audio = req.get("audio") or {}
        wav = audio.get("waveform")
        sr = int(audio.get("sample_rate", self.CAPS.sample_rate))
        n = int(wav.shape[-1]) if wav is not None else 0
        dur = n / sr if sr else 0.0
        text = f"[reference_asr] {dur:.2f}s, {n} samples @ {sr} Hz (no real transcription)"
        return {
            "text": text,
            "segments": [{"start": 0.0, "end": round(dur, 3), "text": text}],
            "language": req.get("language", "auto"),
        }
