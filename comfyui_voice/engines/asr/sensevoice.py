"""SenseVoice-Small (FunASR) — very fast multilingual ASR + emotion/event tags.

Subprocess-isolated (funasr stack). License: custom model license (attribution +
termination clause), review before commercial use. ~15x faster than Whisper-large.
No native word timestamps. Set up runtimes/sensevoice/.venv.
"""
from __future__ import annotations

from typing import Any

from ...base import BaseEngine, EngineCapabilities
from ...registry import register_engine


@register_engine("sensevoice")
class SenseVoiceSmall(BaseEngine):
    CAPS = EngineCapabilities(
        id="sensevoice",
        display_name="SenseVoice-Small (FunASR, fast)",
        tasks=("asr",),
        license="custom (FunASR model license)",
        commercial_safe=False,  # review the model license
        languages=("ko", "zh", "en", "ja", "yue"),
        sample_rate=16000,
        vram_est_gb=2.0,
        isolation="subprocess",
        pip_install=("funasr>=1.1.3", "huggingface_hub", "torchaudio"),
        probe_import=("funasr",),
        param_schema={"params": {}},
    )

    def load(self) -> None:
        from funasr import AutoModel

        self._model = AutoModel(
            model="iic/SenseVoiceSmall",
            trust_remote_code=True,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0",
            disable_update=True,
        )

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        from ...audio_utils import to_mono_np

        audio_np, _ = to_mono_np(req["audio"], 16000)
        lang = req.get("language")
        lang = "auto" if lang in (None, "auto") else lang
        res = self._model.generate(
            input=audio_np, cache={}, language=lang, use_itn=True,
            batch_size_s=60, merge_vad=True, merge_length_s=15,
        )
        text = rich_transcription_postprocess(res[0]["text"])
        return {"text": text, "segments": [{"start": None, "end": None, "text": text}], "language": lang}
