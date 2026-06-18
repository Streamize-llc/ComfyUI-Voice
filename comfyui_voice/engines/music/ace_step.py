"""ACE-Step 1.5 — text-to-music. MIT / commercial-safe. REUSES ComfyUI core.

No vendoring, no extra pip deps: ComfyUI core natively supports ACE-Step 1.5
(comfy_extras/nodes_ace.py, comfy.model_base.ACEStep15, CLIPType.ACE, 48kHz
Oobleck VAE). This adapter just loads the components via comfy.sd and runs the
core sampling pipeline in-process on the host torch. 50+ languages incl. Korean.

Setup (one-time): download the 4 files from HF Comfy-Org/ace_step_1.5_ComfyUI_files
(split_files) into the standard ComfyUI model folders (~13.7GB):
  models/diffusion_models/acestep_v1.5_turbo.safetensors   (4.5GB, DiT)
  models/text_encoders/qwen_0.6b_ace15.safetensors          (1.2GB, code planner)
  models/text_encoders/qwen_4b_ace15.safetensors            (7.9GB, text encoder)
  models/vae/ace_1.5_vae.safetensors                        (0.3GB, audio VAE)
Verified: 10s stereo @48kHz in ~4s on an RTX 4090.
"""
from __future__ import annotations

from typing import Any

from ...base import BaseEngine, EngineCapabilities
from ...registry import register_engine

_LANGS = ("en", "ko", "zh", "ja", "es", "fr", "de", "it", "ru", "pt", "ar", "hi")


@register_engine("ace_step")
class AceStep15(BaseEngine):
    CAPS = EngineCapabilities(
        id="ace_step",
        display_name="ACE-Step 1.5 (music · MIT)",
        tasks=("music",),
        license="Apache-2.0 / MIT",
        commercial_safe=True,
        languages=_LANGS,  # 50+; lyrics drive language
        sample_rate=48000,
        output_channels=2,
        vram_est_gb=14.0,
        isolation="inproc",  # ComfyUI core — host torch
        pip_install=(),  # core covers it; only the 4 model files are needed
        probe_import=("comfy",),
        param_schema={
            "params": {
                "text": {"type": "string", "multiline": True, "default": "lofi hip hop, chill piano, soft drums, 90 bpm", "targets": ["text"]},
                "lyrics": {"type": "string", "multiline": True, "default": "[instrumental]", "targets": ["lyrics"]},
                "duration": {"type": "float", "default": 30.0, "min": 1.0, "max": 240.0, "step": 1.0, "targets": ["duration"]},
                "language": {"type": "enum", "default": "en", "options": list(_LANGS), "targets": ["language"]},
                "bpm": {"type": "int", "default": 120, "min": 10, "max": 300, "targets": ["bpm"]},
                "steps": {"type": "int", "default": 8, "min": 1, "max": 60, "targets": ["steps"]},
                "seed": {"type": "int", "default": 42, "min": 0, "max": 2**32 - 1, "targets": ["seed"]},
            }
        },
    )

    def load(self) -> None:
        import comfy.sd
        import comfy.utils
        import folder_paths

        def need(folder: str, name: str) -> str:
            p = folder_paths.get_full_path(folder, name)
            if p is None:
                raise RuntimeError(
                    f"ACE-Step 1.5: missing {folder}/{name}. Download the 4 split files from "
                    "Comfy-Org/ace_step_1.5_ComfyUI_files (see adapter docstring)."
                )
            return p

        self._model = comfy.sd.load_diffusion_model(need("diffusion_models", "acestep_v1.5_turbo.safetensors"))
        self._clip = comfy.sd.load_clip(
            ckpt_paths=[need("text_encoders", "qwen_0.6b_ace15.safetensors"),
                        need("text_encoders", "qwen_4b_ace15.safetensors")],
            clip_type=comfy.sd.CLIPType.ACE,
        )
        self._vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(need("vae", "ace_1.5_vae.safetensors")))

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        import nodes
        import torch
        import comfy.model_management as mm
        from comfy_extras.nodes_audio import vae_decode_audio
        from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow

        text = str(req.get("text") or "").strip()
        lyrics = str(req.get("lyrics") or "[instrumental]")
        dur = max(1.0, float(req.get("duration", 30.0)))
        seed = int(req.get("seed", 42))
        bpm = int(req.get("bpm", 120))
        steps = int(req.get("steps", 8))
        lang = req.get("language") or "en"
        if lang in ("auto", "any"):
            lang = "en"

        tokens = self._clip.tokenize(
            text, lyrics=lyrics, bpm=bpm, duration=dur, timesignature=4, language=lang,
            keyscale="C major", seed=seed, generate_audio_codes=True,
            cfg_scale=2.0, temperature=0.85, top_p=0.9, top_k=0, min_p=0.0,
        )
        positive = self._clip.encode_from_tokens_scheduled(tokens)
        negative = nodes.ConditioningZeroOut().zero_out(positive)[0]
        length = round(dur * 48000 / 1920)
        latent = {
            "samples": torch.zeros([1, 64, length], device=mm.intermediate_device(), dtype=mm.intermediate_dtype()),
            "type": "audio",
            "downscale_ratio_temporal": 1764,
        }
        patched = ModelSamplingAuraFlow().patch_aura(self._model, 3.0)[0]
        out = nodes.common_ksampler(patched, seed, steps, 1.0, "euler", "simple", positive, negative, latent, denoise=1.0)[0]
        audio = vae_decode_audio(self._vae, out)
        return {"waveform": audio["waveform"], "sample_rate": audio["sample_rate"]}
