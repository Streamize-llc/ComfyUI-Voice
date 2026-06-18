"""WhisperX — ASR + word-level alignment + (optional) speaker diarization. BSD-2 (code).

Subprocess-isolated. Korean alignment uses kresnik/wav2vec2-large-xlsr-korean
(auto-selected). Diarization needs a HF token (pyannote, gated/CC-BY) — set
HF_TOKEN in the engine's venv env to enable it; otherwise ASR+align still work.
Set up runtimes/whisperx/.venv.
"""
from __future__ import annotations

from typing import Any

from ...base import BaseEngine, EngineCapabilities
from ...registry import register_engine


@register_engine("whisperx")
class WhisperX(BaseEngine):
    CAPS = EngineCapabilities(
        id="whisperx",
        display_name="WhisperX (align + diarize)",
        tasks=("asr",),
        license="BSD-2 (code); pyannote gated for diarization",
        commercial_safe=True,
        weights_gated=True,  # diarization model is gated
        languages=("ko", "en", "ja", "zh"),
        sample_rate=16000,
        vram_est_gb=8.0,
        isolation="subprocess",
        pip_install=("whisperx==3.8.6",),
        probe_import=("whisperx",),
        param_schema={"params": {}},
    )

    def load(self) -> None:
        import os

        import torch
        import whisperx

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        self._wx = whisperx
        self._device = device
        self._asr = whisperx.load_model("large-v3", device, compute_type=compute, language="ko")
        self._align, self._meta = whisperx.load_align_model(language_code="ko", device=device)
        self._diar = None
        token = os.environ.get("HF_TOKEN")
        if token:
            from whisperx.diarize import DiarizationPipeline

            self._diar = DiarizationPipeline(token=token, device=device)

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        from ...audio_utils import to_mono_np

        audio_np, _ = to_mono_np(req["audio"], 16000)
        result = self._asr.transcribe(audio_np, batch_size=16)
        result = self._wx.align(
            result["segments"], self._align, self._meta, audio_np, self._device,
            return_char_alignments=False,
        )
        if self._diar is not None:
            diarize_df = self._diar(audio_np)
            result = self._wx.assign_word_speakers(diarize_df, result)
        text = " ".join(s["text"].strip() for s in result["segments"]).strip()
        return {"text": text, "segments": result["segments"], "language": "ko"}
