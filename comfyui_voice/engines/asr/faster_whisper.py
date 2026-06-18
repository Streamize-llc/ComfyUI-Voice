"""faster-whisper (large-v3-turbo) — fast workhorse ASR. MIT.

In-process and SAFE for the host stack: CTranslate2 is independent of PyTorch, so
it never moves torch/transformers/numpy. Word/segment timestamps supported.
"""
from __future__ import annotations

from typing import Any

from ...base import BaseEngine, EngineCapabilities
from ...registry import register_engine


@register_engine("faster_whisper")
class FasterWhisper(BaseEngine):
    MODEL_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"

    CAPS = EngineCapabilities(
        id="faster_whisper",
        display_name="faster-whisper (large-v3-turbo)",
        tasks=("asr",),
        license="MIT",
        commercial_safe=True,
        languages=("ko", "en", "ja", "zh"),
        sample_rate=16000,
        vram_est_gb=6.0,
        isolation="inproc",
        pip_install=("faster-whisper>=1.1.0",),
        probe_import=("faster_whisper",),
        param_schema={"params": {}},
    )

    def load(self) -> None:
        import faster_whisper
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        try:
            self._model = faster_whisper.WhisperModel(self.MODEL_ID, device=device, compute_type=compute)
        except Exception:  # cuDNN/cuBLAS DLL issues on Windows -> CPU fallback
            self._model = faster_whisper.WhisperModel(self.MODEL_ID, device="cpu", compute_type="int8")

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        from ...audio_utils import to_mono_np

        audio_np, _ = to_mono_np(req["audio"], 16000)
        lang = req.get("language")
        lang = None if lang in (None, "auto") else lang
        word_ts = bool(req.get("word_timestamps", False))

        segments, info = self._model.transcribe(
            audio_np, language=lang, task="transcribe", word_timestamps=word_ts, beam_size=5,
        )
        out: list[dict] = []
        for seg in segments:  # lazy generator — iterating runs inference
            entry = {"start": seg.start, "end": seg.end, "text": seg.text}
            if word_ts and seg.words:
                entry["words"] = [
                    {"start": w.start, "end": w.end, "word": w.word} for w in seg.words
                ]
            out.append(entry)
        return {
            "text": "".join(e["text"] for e in out),
            "segments": out,
            "language": info.language,
        }
