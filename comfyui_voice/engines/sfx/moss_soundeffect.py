"""MOSS-SoundEffect v2.0 — text-to-sound-effect / Foley. Apache-2.0, commercial-safe.

OpenMOSS's text-to-audio diffusion pipeline (DiT + DAC VAE + Qwen3 text encoder +
flow-match scheduler). Runs NATIVELY on ComfyUI's host torch — no separate torch,
no `descript-audiotools` (vendored a 50-line BaseModel.load shim instead), no
`AutoencoderOobleck` (the DAC VAE is used exclusively). Upstream inference code is
vendored under `comfyui_voice/_vendor/moss_soundeffect_v2` (credit + Apache-2.0).

Setup (one-time): download the full HF repo `fnlp/MOSS-SoundEffect-v2.0` (~11GB,
diffusers-style: model_index.json + transformer/ vae/ text_encoder/ tokenizer/
scheduler/) into ComfyUI's `models/audio/MOSS-SoundEffect-v2.0/`.

Note: the pipeline always denoises a fixed 30s latent and crops to `duration`, so
wall-clock is governed by `steps` (× 2 for CFG), not by the requested duration.
Verified: 48kHz mono SFX on an RTX 4090.
"""
from __future__ import annotations

import os
from typing import Any

from ...base import BaseEngine, EngineCapabilities
from ...paths import SUITE_ROOT
from ...registry import register_engine

# torch.compile + Triton cudagraphs (decorating model_fn_wan_video upstream) are
# brittle on Windows; force eager. Set before the vendored package is imported.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

_VENDOR = os.path.join(SUITE_ROOT, "comfyui_voice", "_vendor")
_DEFAULT_SUBDIR = os.path.join("audio", "MOSS-SoundEffect-v2.0")


def _resolve_model_dir() -> str:
    """Locate the MOSS weights under ComfyUI's models/ tree."""
    try:
        import folder_paths

        cand = os.path.join(folder_paths.models_dir, _DEFAULT_SUBDIR)
        if os.path.isdir(cand):
            return cand
    except Exception:
        pass
    # Fallback: ComfyUI root inferred from SUITE_ROOT (.../custom_nodes/ComfyUI-Voice).
    comfy_root = os.path.dirname(os.path.dirname(SUITE_ROOT))
    return os.path.join(comfy_root, "models", _DEFAULT_SUBDIR)


@register_engine("moss_soundeffect")
class MossSoundEffect(BaseEngine):
    CAPS = EngineCapabilities(
        id="moss_soundeffect",
        display_name="MOSS-SoundEffect v2.0 (sfx · Apache-2.0)",
        tasks=("sfx",),
        version="2.0",
        license="Apache-2.0",
        commercial_safe=True,
        languages=("en",),  # English natural-language sound descriptions
        sample_rate=48000,
        output_channels=1,
        vram_est_gb=12.0,
        model_size_gb=11.0,
        isolation="inproc",  # host torch
        pip_install=("diffusers", "ftfy", "regex"),
        probe_import=("diffusers", "ftfy", "regex"),
        param_schema={
            "params": {
                "text": {"type": "string", "multiline": True, "default": "heavy rain on a tin roof with distant thunder", "targets": ["text"]},
                "duration": {"type": "float", "default": 10.0, "min": 0.5, "max": 30.0, "step": 0.5, "targets": ["duration"]},
                "seed": {"type": "int", "default": 42, "min": 0, "max": 2**32 - 1, "targets": ["seed"]},
                "steps": {"type": "int", "default": 100, "min": 10, "max": 200, "targets": ["steps"]},
                "cfg_scale": {"type": "float", "default": 4.0, "min": 1.0, "max": 12.0, "step": 0.5, "targets": ["cfg_scale"]},
                "negative_prompt": {"type": "string", "multiline": True, "default": "", "targets": ["negative_prompt"]},
            }
        },
    )

    def load(self) -> None:
        import sys

        import torch

        # Belt-and-suspenders: disable dynamo even if torch was imported before
        # the env var above took effect.
        try:
            import torch._dynamo

            torch._dynamo.config.disable = True
        except Exception:
            pass

        model_dir = _resolve_model_dir()
        if not os.path.isdir(model_dir):
            raise RuntimeError(
                f"MOSS-SoundEffect: weights not found at {model_dir}. Download the HF repo "
                "fnlp/MOSS-SoundEffect-v2.0 (~11GB) into models/audio/MOSS-SoundEffect-v2.0/."
            )

        if _VENDOR not in sys.path:
            sys.path.insert(0, _VENDOR)
        from moss_soundeffect_v2 import MossSoundEffectPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipe = MossSoundEffectPipeline.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, device=device
        )
        self._loaded = True

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        import torch

        text = str(req.get("text") or "").strip()
        if not text:
            raise ValueError("MOSS-SoundEffect: empty prompt.")
        dur = max(0.5, min(float(req.get("duration", 10.0)), 30.0))
        seed = int(req.get("seed", 42))
        steps = int(req.get("steps", 100))
        cfg = float(req.get("cfg_scale", 4.0))
        neg = str(req.get("negative_prompt") or "")

        audio = self._pipe(
            prompt=text,
            seconds=dur,
            num_inference_steps=steps,
            cfg_scale=cfg,
            sigma_shift=5.0,
            seed=seed,
            negative_prompt=neg,
        )
        # audio: (B, C, T) bfloat16 on device -> float32 CPU for the AUDIO type.
        wav = audio.detach().to("cpu", dtype=torch.float32)
        return {"waveform": wav, "sample_rate": 48000}
