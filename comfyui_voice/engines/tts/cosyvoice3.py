"""CosyVoice 3.0 (0.5B) — SOTA zero-shot cloning TTS, native Korean. Apache-2.0.

Subprocess-isolated: the CosyVoice repo pins torch==2.3.1 / transformers==4.51.3
/ numpy==1.26.4 and is a CLONED REPO (not a pip package). Set up its venv at
runtimes/cosyvoice3/.venv, clone https://github.com/FunAudioLLM/CosyVoice
(--recursive), and point env vars at it:
  COSYVOICE_REPO  = path to the cloned repo
  COSYVOICE3_DIR  = path to the downloaded Fun-CosyVoice3-0.5B weights
Every render needs a reference/prompt voice (no built-in speakers).
"""
from __future__ import annotations

from typing import Any

from ...base import BaseEngine, EngineCapabilities
from ...registry import register_engine


@register_engine("cosyvoice3")
class CosyVoice3(BaseEngine):
    CAPS = EngineCapabilities(
        id="cosyvoice3",
        display_name="CosyVoice 3.0 (0.5B, clone)",
        tasks=("tts", "voice_conversion"),
        license="Apache-2.0",
        commercial_safe=True,
        supports_cloning=True,
        needs_ref_audio=True,
        languages=("ko", "zh", "en", "ja", "de", "es", "fr", "it", "ru"),
        sample_rate=24000,
        vram_est_gb=4.0,
        isolation="subprocess",
        max_input_chars=2000,
        dep_pins={"torch": "==2.3.1", "transformers": "==4.51.3", "numpy": "==1.26.4"},
        pip_install=("git+https://github.com/FunAudioLLM/CosyVoice.git",),
        probe_import=("cosyvoice.cli.cosyvoice",),
        param_schema={
            "params": {
                "text": {"type": "string", "multiline": True, "default": "안녕하세요, 만나서 반갑습니다.", "targets": ["text"]},
                "speed": {"type": "float", "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "targets": ["speed"]},
            }
        },
    )

    def load(self) -> None:
        import os
        import sys

        repo = os.environ["COSYVOICE_REPO"]
        sys.path.insert(0, repo)
        sys.path.insert(0, os.path.join(repo, "third_party", "Matcha-TTS"))
        from cosyvoice.cli.cosyvoice import AutoModel
        from cosyvoice.utils.file_utils import load_wav

        self._load_wav = load_wav
        model_dir = os.environ.get("COSYVOICE3_DIR", "pretrained_models/Fun-CosyVoice3-0.5B")
        self._model = AutoModel(model_dir=model_dir, fp16=True)
        self._sr = int(self._model.sample_rate)

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        import torch

        from ...audio_utils import write_temp_wav

        text = str(req.get("text") or "").strip()
        ref = req.get("ref_audio")
        if not ref:
            raise ValueError("CosyVoice3 requires a reference/prompt audio (connect voice_ref).")
        prompt = self._load_wav(write_temp_wav(ref), 16000)
        chunks = [
            out["tts_speech"]
            for out in self._model.inference_cross_lingual(
                text, prompt, zero_shot_spk_id="", stream=False,
                speed=float(req.get("speed", 1.0)), text_frontend=True,
            )
        ]
        waveform = torch.cat(chunks, dim=1) if chunks else torch.zeros(1, 1)
        return {"waveform": waveform, "sample_rate": self._sr}
