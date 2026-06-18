"""MMS-TTS Korean (VITS via transformers) — NATIVE on our stack.

Reference implementation of the "native" tier: the model layers run on the
HOST torch 2.12 through transformers' VitsModel (transformers is the architecture
library already in our stack, like comfy/ is for diffusion) — no upstream pip
package, no version pins, no subprocess. VITS is the same architecture family as
MeloTTS, so this doubles as the porting blueprint for a commercial-safe
MeloTTS-Korean native port.

License: facebook/mms-tts-kor is CC-BY-NC-4.0 (non-commercial) — use for
research/eval; not for the commercial product (commercial_safe=False).
"""
from __future__ import annotations

from typing import Any

from ...base import BaseEngine, EngineCapabilities
from ...registry import register_engine


@register_engine("mms_tts_korean")
class MMSTTSKorean(BaseEngine):
    MODEL_ID = "facebook/mms-tts-kor"

    CAPS = EngineCapabilities(
        id="mms_tts_korean",
        display_name="MMS-TTS Korean (VITS · native on torch 2.12)",
        tasks=("tts",),
        license="CC-BY-NC-4.0",
        commercial_safe=False,  # non-commercial weights — eval/reference only
        supports_cloning=False,
        languages=("ko",),
        sample_rate=16000,
        vram_est_gb=0.5,
        isolation="inproc",  # runs on host torch via transformers — no isolation needed
        pip_install=("uroman",),  # transformers + torch already present; uroman = KO romanizer frontend
        probe_import=("transformers", "torch", "uroman"),
        param_schema={
            "params": {
                "text": {"type": "string", "multiline": True, "default": "안녕하세요. 음성 합성 테스트입니다.", "targets": ["text"]},
                "speed": {"type": "float", "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "targets": ["speed"]},
            }
        },
    )

    def load(self) -> None:
        import torch
        from transformers import AutoTokenizer, VitsModel

        self._model = VitsModel.from_pretrained(self.MODEL_ID)
        self._tok = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device).eval()
        self._sr = int(self._model.config.sampling_rate)

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        import torch

        text = str(req.get("text") or "").strip()
        speed = float(req.get("speed", 1.0)) or 1.0
        # VITS: speaking_rate is duration scale (higher = slower); invert speed.
        try:
            self._model.speaking_rate = 1.0 / speed
        except Exception:
            pass
        inputs = self._tok(text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            waveform = self._model(**inputs).waveform  # [1, T] float32
        return {"waveform": waveform.squeeze(0).float().cpu(), "sample_rate": self._sr}
