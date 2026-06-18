"""Qwen3-ASR (1.7B) — flagship native-Korean ASR + forced-aligner timestamps. Apache-2.0.

Subprocess-isolated (qwen-asr stack can move transformers). Loads
Qwen3-ForcedAligner-0.6B for word/char timestamps (extra ~1.3GB VRAM). Set up
runtimes/qwen3_asr/.venv.
"""
from __future__ import annotations

from typing import Any

from ...base import BaseEngine, EngineCapabilities
from ...registry import register_engine


@register_engine("qwen3_asr")
class Qwen3ASR(BaseEngine):
    CAPS = EngineCapabilities(
        id="qwen3_asr",
        display_name="Qwen3-ASR 1.7B (+aligner)",
        tasks=("asr",),
        license="Apache-2.0",
        commercial_safe=True,
        languages=("ko", "en", "zh", "ja"),
        sample_rate=16000,
        vram_est_gb=6.0,
        isolation="subprocess",
        pip_install=("qwen-asr==0.0.6",),
        probe_import=("qwen_asr",),
        param_schema={"params": {}},
    )

    def load(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        self._model = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-1.7B",
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_new_tokens=256,
            forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
            forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0"),
        )

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        from ...audio_utils import to_mono_np

        audio_np, sr = to_mono_np(req["audio"], 16000)
        lang = req.get("language")
        lang = "Korean" if lang in (None, "auto", "ko") else lang
        results = self._model.transcribe(
            audio=(audio_np, sr), language=lang, return_time_stamps=bool(req.get("word_timestamps", True))
        )
        r = results[0]
        segments = []
        for ts in (getattr(r, "time_stamps", None) or []):
            segments.append({"text": ts.text, "start": ts.start_time, "end": ts.end_time})
        return {"text": r.text, "segments": segments, "language": getattr(r, "language", lang)}
