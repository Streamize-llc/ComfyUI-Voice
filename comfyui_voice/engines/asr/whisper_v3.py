"""Whisper large-v3 via transformers pipeline — multilingual ASR. Apache-2.0.

In-process: uses the HOST transformers (5.10.x) + torch, so no extra heavy deps
beyond `accelerate`. Heavier VRAM than faster-whisper; weights auto-download.
"""
from __future__ import annotations

from typing import Any

from ...base import BaseEngine, EngineCapabilities
from ...registry import register_engine


@register_engine("whisper_v3")
class WhisperV3(BaseEngine):
    CAPS = EngineCapabilities(
        id="whisper_v3",
        display_name="Whisper large-v3 (transformers)",
        tasks=("asr",),
        license="Apache-2.0",
        commercial_safe=True,
        languages=("ko", "en", "ja", "zh"),
        sample_rate=16000,
        vram_est_gb=10.0,
        isolation="inproc",
        pip_install=("accelerate",),
        probe_import=("transformers", "torch"),
        param_schema={"params": {}},
    )

    def load(self) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "openai/whisper-large-v3", torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=device,
        )

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        from ...audio_utils import to_mono_np

        audio_np, sr = to_mono_np(req["audio"], 16000)
        lang = req.get("language")
        gen_kwargs: dict[str, Any] = {"task": "transcribe"}
        if lang and lang != "auto":
            gen_kwargs["language"] = lang
        result = self._pipe(
            {"raw": audio_np, "sampling_rate": sr},
            chunk_length_s=30,
            batch_size=8,
            return_timestamps=True,
            generate_kwargs=gen_kwargs,
        )
        segments = [
            {"start": c["timestamp"][0], "end": c["timestamp"][1], "text": c["text"]}
            for c in result.get("chunks", [])
        ]
        return {"text": result["text"], "segments": segments, "language": lang or "auto"}
