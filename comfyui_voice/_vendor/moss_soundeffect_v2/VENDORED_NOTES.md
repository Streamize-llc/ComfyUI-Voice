# Vendored: MOSS-SoundEffect v2.0 (inference subset)

Source: https://github.com/OpenMOSS/MOSS-TTS — `moss_soundeffect_v2/` package.
License: Apache-2.0 (see `LICENSE_MOSS_SoundEffect`).

This is a **trimmed, patched** copy carried so the engine runs natively on
ComfyUI's host torch (no separate torch, no `descript-audiotools`, no env churn).

## What was kept
`__init__.py`, `pipeline_moss_soundeffect.py`, and the `diffsynth/` inference
tree: `models/` (dac_vae, qwen3_text_encoder, wan_audio_dit, wan_video_dit,
wan_video_camera_controller, utils), `pipelines/wan_audio`, `prompters/`,
`schedulers/flow_match`, `utils/` (BasePipeline).

## What was dropped (not on the text-to-audio generate path)
- `diffsynth/trainers/` — pulls `torchcodec` (and training-only code).
- `finetuning/`, `hf_export.py`, `infer_from_pipeline.*`, `pyproject.toml` — tooling/demo.

## Patches applied (search for "Vendored ComfyUI-Voice")
1. `diffsynth/models/dac_vae.py` — `from audiotools …` → `from ._audiotools_compat …`.
   `_audiotools_compat.py` supplies a ~50-line `BaseModel.load` (reconstructs the
   DAC from a `{"state_dict","metadata":{"kwargs"}}` checkpoint) and an
   `AudioSignal` placeholder. The generate/decode path never needs real audiotools.
2. `diffsynth/pipelines/wan_audio.py` — `from diffusers import AutoencoderOobleck`
   wrapped in try/except (diffusers <0.30 lacks it; the DAC VAE is used exclusively).

## Runtime notes
- `model_fn_wan_video` is decorated with `@torch.compile(... triton.cudagraphs ...)`;
  the adapter sets `TORCHDYNAMO_DISABLE=1` (and `torch._dynamo.config.disable`) so it
  runs eager — robust on Windows / without Triton.
- Attention falls back to `F.scaled_dot_product_attention` when flash/sage attn
  are absent (they are, here).
- Verified on transformers 5.10 / diffusers 0.29 / torch 2.12 (RTX 4090): the DiT
  loads with 0 missing/unexpected keys; 48 kHz mono SFX, prompt-conditioned.
