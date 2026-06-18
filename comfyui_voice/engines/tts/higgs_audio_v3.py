"""Higgs Audio v3 (4B) — expressive multilingual TTS + zero-shot clone. EVAL-ONLY.

⚠️ NON-COMMERCIAL: "Boson Higgs Audio v3 Research and Non-Commercial License" —
production / hosted / revenue use requires a separate commercial license. Shipped
here as eval/reference only (commercial_safe=False), like mms_tts_korean.

NATIVE on the host torch 2.12 / transformers 5.10 — no separate torch, no
subprocess. The model architecture (higgs_multimodal_qwen3) is not in any
transformers release yet, but the MIT-licensed native reimplementation from
Saganaki22/Higgs_v3-TTS-ComfyUI (built for Transformers 5.3–5.5) runs cleanly on
5.10 — vendored under comfyui_voice/_vendor/higgs_native (credit + LICENSE there).
The audio codec is transformers' own HiggsAudioV2Tokenizer (present in 5.10).

Setup (one-time): download the weights to
  <ComfyUI>/models/higgsv3tts/higgs-audio-v3-tts-4b/
(HF: bosonai/higgs-audio-v3-tts-4b, ~9.3GB; config/tokenizer/index come with it).
Verified: Korean default-voice TTS round-trips near-perfectly on an RTX 4090.
"""
from __future__ import annotations

import os
from typing import Any

from ...base import BaseEngine, EngineCapabilities
from ...paths import PKG_DIR
from ...registry import register_engine

_VENDOR = os.path.join(PKG_DIR, "_vendor")


@register_engine("higgs_audio_v3")
class HiggsAudioV3(BaseEngine):
    CAPS = EngineCapabilities(
        id="higgs_audio_v3",
        display_name="Higgs Audio v3 (4B · clone · eval-only)",
        tasks=("tts",),
        license="Boson Higgs Audio v3 Research & Non-Commercial",
        commercial_safe=False,  # NON-COMMERCIAL — eval/reference only
        supports_cloning=True,
        needs_ref_audio=False,  # default voice when no reference
        supports_emotion=True,
        languages=("ko", "en", "zh", "ja", "de", "es", "fr", "it", "ru", "pt"),  # 100+ total
        sample_rate=24000,
        vram_est_gb=12.0,
        isolation="inproc",
        max_input_chars=2000,
        pip_install=("accelerate",),
        probe_import=("transformers", "torch"),
        param_schema={
            "params": {
                "text": {"type": "string", "multiline": True, "default": "안녕하세요. 반갑습니다.", "targets": ["text"]},
                "temperature": {"type": "float", "default": 0.7, "min": 0.1, "max": 1.5, "step": 0.05, "targets": ["temperature"]},
                "top_p": {"type": "float", "default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05, "targets": ["top_p"]},
                "seed": {"type": "int", "default": 42, "min": 0, "max": 2**32 - 1, "targets": ["seed"]},
            }
        },
    )

    def load(self) -> None:
        import sys

        if _VENDOR not in sys.path:
            sys.path.insert(0, _VENDOR)
        from higgs_native.loader import load_higgs_bundle
        from higgs_native.native import generate_higgs_audio

        self._generate = generate_higgs_audio
        # weights expected at <ComfyUI>/models/higgsv3tts/higgs-audio-v3-tts-4b/
        self._bundle = load_higgs_bundle(
            "higgs-audio-v3-tts-4b", "bf16", "cuda", "sdpa", download_if_missing=False
        )

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        out = self._generate(
            self._bundle,
            text=str(req.get("text") or "").strip(),
            reference_audio=req.get("ref_audio"),  # VOICE_REF audio for zero-shot clone
            reference_audio_path="",
            reference_text=str(req.get("ref_text") or ""),
            max_new_tokens=int(req.get("max_new_tokens", 2048)),
            temperature=float(req.get("temperature", 0.7)),
            top_p=float(req.get("top_p", 0.95)),
            top_k=int(req.get("top_k", 50)),
            seed=int(req.get("seed", 42)),
            trim_reference_audio=True,
            silence_threshold_db=-42.0,
            max_reference_seconds=100.0,
        )
        return {"waveform": out["waveform"], "sample_rate": out["sample_rate"]}
