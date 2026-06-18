# ComfyUI-Voice 🎙️

An **extensible, open-source audio/voice node suite** for ComfyUI. The goal is to
own the (currently empty) space of a *proper, unified* audio toolkit — TTS,
voice cloning, voice conversion, ASR, source separation, enhancement, music/SFX
generation and more — under one coherent, contributor-friendly architecture.

> **One mantra:** *one generic node per task + single-file engine adapter +
> capability declaration.* Adding a model = drop one file in
> `comfyui_voice/engines/<task>/`. Nothing else changes.

## Supported models

Everything runs **natively on the host stack** (torch 2.12 / transformers 5.10 /
numpy 2.4) — no separate torch, no subprocess. Status is honest: ✅ = real
inference verified on our machine (Korean round-trip checked); 🧩 = adapter
included but not yet installed/verified.

### ✅ Verified working (Korean)

| Engine (`id`) | Task | License | Notes |
|---|---|---|---|
| **MeloTTS Korean** (`melotts_korean`) | TTS (preset voice) | **MIT** ✅ | Primary commercial-safe KO TTS. Native (vendored VITS); fast. Round-trip near-perfect. |
| **CosyVoice 3.0** (`cosyvoice3`) | TTS zero-shot **voice clone** + VC | **Apache-2.0** ✅ | SOTA cloning, native KO/ZH/EN/JA… Needs one-time setup (repo + weights, see below). |
| **Supertonic 3** (`supertonic`) | TTS (preset, on-device) | code MIT / model **OpenRAIL-M** ⚠️ | ONNX (truly torch-independent, cleanest install). Native KO. Use `voice="F1"`, `total_steps≥16` (M-voices/low steps garble). Review OpenRAIL-M before commercial use. |
| **MMS-TTS Korean** (`mms_tts_korean`) | TTS (preset) | CC-BY-NC ⚠️ | Via `transformers` VitsModel. **Non-commercial — eval/reference only.** |
| **faster-whisper** (`faster_whisper`) | ASR (STT) | **MIT** ✅ | large-v3-turbo, CTranslate2 (torch-independent). Word timestamps. |
| **Whisper large-v3** (`whisper_v3`) | ASR (STT) | **Apache-2.0** ✅ | Via host `transformers`. `VOICE_WHISPER_MODEL` env selects model size. |

`korean_whisper` (Apache-2.0, ghost613 KO fine-tune on the faster-whisper
backend) is included and shares the verified `faster_whisper` code path; it
downloads its model on first use and hasn't been separately benchmarked here.

### 🧩 Adapter included, not yet verified

Drop-in adapters exist (capability-declared, appear in the dropdown) but their
model deps aren't installed/tested yet — enable per the `Voice Engine Info`
node's install hint, then verify:

- **TTS:** `kokoro` (Apache, **no Korean**), `chatterbox` (MIT, clone+emotion),
  `qwen3_tts` (Apache, clone), `oute_tts` (Apache).
- **ASR:** `qwen3_asr` (Apache, +timestamps), `sensevoice` (custom ⚠️, fast),
  `whisperx` (BSD, alignment + diarization).
- **Demo (no deps):** `reference_tone` (TTS), `reference_asr` — for testing the
  pipeline on a clean install and as the adapter template.

### CosyVoice 3.0 one-time setup

Native on torch 2.12 with three load()-time patches (torchcodec-free `load_wav`,
transformers-5.x KV-cache mask, LLM float32 cast — all in the adapter):

```bash
# 1) clone the inference code (kept under runtimes/, gitignored)
git clone --recursive https://github.com/FunAudioLLM/CosyVoice \
  custom_nodes/ComfyUI-Voice/comfyui_voice/runtimes/cosyvoice3/CosyVoice
# 2) weights -> models/voice/Fun-CosyVoice3-0.5B (HF FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
# 3) deps (dry-run-clean on the pinned stack):
pip install hyperpyyaml x-transformers einops inflect onnxruntime wetext \
  modelscope conformer diffusers lightning gdown wget pyarrow pyworld librosa
pip install --no-build-isolation openai-whisper==20231117
```

## Two adapter tiers (the dependency strategy)

Following ComfyUI core's philosophy (it reimplements model architectures in
`comfy/` rather than pip-installing diffusers), engines come in two tiers:

1. **Native (preferred, `isolation="inproc"`):** the model runs on the HOST
   torch 2.12 — via `transformers` (the architecture lib already in our stack,
   e.g. `mms_tts_korean`, `whisper_v3`) or by vendoring the official inference
   code with small pure-python deps + load()-time patches (e.g. `melotts_korean`,
   `cosyvoice3`). No version pins, no dependency conflicts, no subprocess.
   **This is the target for every engine** and how all verified models run today.
2. **Thin-wrapper (fallback, `isolation="subprocess"`):** pip-install the
   upstream package into a per-engine venv and drive it over the worker protocol.
   Scaffolded for not-yet-ported engines; none are required by the verified set.

## Design principles

1. **Composable via typed sockets.** Everything speaks the core `AUDIO` dict
   (`{"waveform": Tensor[B,C,T], "sample_rate": int}`) plus a few `VOICE_*`
   custom types, so chains interoperate with core LoadAudio/SaveAudio for free.
2. **Engine = single-file drop-in adapter.** Self-registered via
   `@register_engine`; no central dispatch to edit.
3. **Capability-driven everything.** One `EngineCapabilities` dataclass per
   engine generates the form, the validator and (later) the MCP tool — never
   branch on the engine id.
4. **Dependency isolation is a declared field.** `isolation="inproc"` or
   `"subprocess"` (own venv) — conflicting model pins coexist without a rewrite.
5. **Reuse core, never reinvent.** Save/preview/encode come from ComfyUI core.

## Custom socket types

`AUDIO` is reused unprefixed (interop). All suite-private types carry the
`VOICE_` prefix and are declared in `comfyui_voice/types.py` (`RESERVED_TYPES`).
**Type strings and `node_id`s are a frozen, append-only API** — never rename one
(it breaks saved workflows); deprecate via alias.

## Adding an engine (the whole contract)

Copy `comfyui_voice/engines/tts/_reference_tone.py` to
`comfyui_voice/engines/tts/<name>.py` and:

```python
@register_engine("my_tts")
class MyTTS(BaseEngine):
    CAPS = EngineCapabilities(
        id="my_tts", display_name="My TTS",
        tasks=("tts",), license="Apache-2.0", commercial_safe=True,
        supports_cloning=True, needs_ref_audio=True,
        languages=("ko", "en"), sample_rate=24000, vram_est_gb=6.0,
        isolation="inproc",          # or "subprocess" if its pins conflict
        param_schema={"params": {...}},
    )
    def load(self):
        import my_model            # lazy-import heavy deps HERE
        self.m = my_model.load(...)
    def generate(self, task, req):
        wav, sr = self.m.tts(req["text"], ...)
        return {"waveform": wav, "sample_rate": sr}
    def unload(self):
        del self.m
```

That's it — the engine appears in the `Voice TTS` dropdown, the form adapts to
its `param_schema`, and validation honors its capabilities. Run the parity check
(`tests/`) before opening a PR.

⚠️ Before adding an engine's deps to the main env, run
`pip install --dry-run` and confirm it does **not** move ComfyUI's pinned
`torch` / `transformers` / `numpy`. If it would, declare `isolation="subprocess"`.

## Tests

```bash
& "C:\Users\qkrwn\anaconda3\envs\ComfyUI\python.exe" custom_nodes/ComfyUI-Voice/tests/test_core.py
```

Pure-core: no ComfyUI server, no model weights. License: Apache-2.0.
