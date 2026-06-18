"""CosyVoice 3.0 (0.5B) — SOTA zero-shot cloning TTS, native Korean. Apache-2.0.

NATIVE on the host torch 2.12 / transformers 5.10 / numpy 2.4 — NO separate
torch, NO subprocess. The upstream repo pins torch==2.3.1/transformers==4.51.3
but it runs on our stack with three patches applied at load():
  1. load_wav -> soundfile (torch 2.12 routes torchaudio.load through torchcodec)
  2. Qwen2Encoder.forward_one_step -> full-length attention_mask (transformers 5.x
     KV-cache needs past+current mask; 4.51 was lenient)
  3. cast the Qwen2 LLM to float32 (weights are bf16, input embeds are f32)

Setup (one-time, kept under runtimes/ which is gitignored):
  - clone https://github.com/FunAudioLLM/CosyVoice (--recursive) to
    runtimes/cosyvoice3/CosyVoice
  - weights at models/voice/Fun-CosyVoice3-0.5B (HF FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
  - deps (dry-run-clean, installed in the conda env): hyperpyyaml x-transformers
    einops inflect onnxruntime wetext modelscope conformer diffusers lightning
    gdown wget pyarrow pyworld librosa openai-whisper

Every render needs a reference voice (zero-shot clone). Verified: Korean
zero-shot round-trips near-perfectly.
"""
from __future__ import annotations

import os
from typing import Any

from ...base import BaseEngine, EngineCapabilities
from ...paths import models_dir, runtimes_dir
from ...registry import register_engine

_REPO = os.path.join(runtimes_dir(), "cosyvoice3", "CosyVoice")
_SYS_PROMPT = "You are a helpful assistant.<|endofprompt|>"


def _patch_transformers5(repo: str) -> None:
    """Apply the transformers-5.x / torch-2.12 compatibility patches at runtime."""
    import torch

    # load_wav: avoid torchaudio.load -> torchcodec; use soundfile.
    from cosyvoice.cli import frontend as _frontend

    def _load_wav(wav, target_sr, min_sr=16000):
        import soundfile as sf

        data, sr = sf.read(wav, dtype="float32", always_2d=True)  # [T, C]
        speech = torch.from_numpy(data.T).mean(dim=0, keepdim=True)  # [1, T]
        if sr != target_sr:
            import torchaudio

            speech = torchaudio.functional.resample(speech, sr, target_sr)
        return speech

    _frontend.load_wav = _load_wav

    # forward_one_step: full-length attention_mask for the cached Qwen2 LLM.
    from cosyvoice.llm.llm import Qwen2Encoder

    def _forward_one_step(self, xs, masks, cache=None):
        try:
            past_len = cache.get_seq_length() if cache is not None else 0
        except Exception:
            past_len = 0
        attn = torch.ones((xs.size(0), past_len + xs.size(1)), dtype=torch.long, device=xs.device)
        outs = self.model(
            inputs_embeds=xs, attention_mask=attn, output_hidden_states=True,
            return_dict=True, use_cache=True, past_key_values=cache,
        )
        return outs.hidden_states[-1], outs.past_key_values

    Qwen2Encoder.forward_one_step = _forward_one_step


@register_engine("cosyvoice3")
class CosyVoice3(BaseEngine):
    CAPS = EngineCapabilities(
        id="cosyvoice3",
        display_name="CosyVoice 3.0 (0.5B, zero-shot clone)",
        tasks=("tts", "voice_conversion"),
        license="Apache-2.0",
        commercial_safe=True,
        supports_cloning=True,
        needs_ref_audio=True,
        languages=("ko", "zh", "en", "ja", "de", "es", "fr", "it", "ru"),
        sample_rate=24000,
        vram_est_gb=4.0,
        isolation="inproc",  # native on host torch 2.12 (with load() patches)
        max_input_chars=2000,
        pip_install=(
            "hyperpyyaml", "x-transformers", "einops", "inflect", "onnxruntime",
            "wetext", "modelscope", "conformer", "diffusers", "lightning",
            "gdown", "wget", "pyarrow", "pyworld", "librosa", "openai-whisper",
        ),
        probe_import=("hyperpyyaml", "onnxruntime", "wetext", "conformer"),
        param_schema={
            "params": {
                "text": {"type": "string", "multiline": True, "default": "안녕하세요, 만나서 반갑습니다.", "targets": ["text"]},
                "speed": {"type": "float", "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "targets": ["speed"]},
            }
        },
    )

    def load(self) -> None:
        import sys

        if not os.path.isdir(_REPO):
            raise RuntimeError(
                f"CosyVoice repo not found at {_REPO}. Clone it (--recursive) there first."
            )
        for p in (_REPO, os.path.join(_REPO, "third_party", "Matcha-TTS")):
            if p not in sys.path:
                sys.path.insert(0, p)
        _patch_transformers5(_REPO)
        from cosyvoice.cli.cosyvoice import AutoModel

        self._model = AutoModel(model_dir=models_dir("Fun-CosyVoice3-0.5B"), fp16=False)
        self._model.model.llm = self._model.model.llm.float()  # bf16 -> f32
        self._sr = int(self._model.sample_rate)

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        import torch

        from ...audio_utils import write_temp_wav

        text = str(req.get("text") or "").strip()
        speed = float(req.get("speed", 1.0)) or 1.0
        ref = req.get("ref_audio")
        if not ref:
            raise ValueError("CosyVoice3 requires a reference/prompt audio (connect voice_ref).")
        ref_path = write_temp_wav(ref)
        ref_text = str(req.get("ref_text") or "").strip()

        if ref_text:  # zero-shot clone (needs the prompt transcript)
            gen = self._model.inference_zero_shot(
                text, _SYS_PROMPT + ref_text, ref_path, stream=False, speed=speed,
            )
        else:  # cross-lingual clone (no transcript needed)
            gen = self._model.inference_cross_lingual(
                _SYS_PROMPT + text, ref_path, stream=False, speed=speed,
            )
        chunks = [out["tts_speech"] for out in gen]
        waveform = torch.cat(chunks, dim=1) if chunks else torch.zeros(1, 1)
        return {"waveform": waveform, "sample_rate": self._sr}
