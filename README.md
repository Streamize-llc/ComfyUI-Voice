<div align="center">

# ComfyUI-Voice 🎙️

**The unified, extensible audio & voice suite for ComfyUI.**
TTS · voice cloning · speech‑to‑text — many state‑of‑the‑art open models, one clean node graph.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-custom%20node-8A2BE2.svg)](https://github.com/comfyanonymous/ComfyUI)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#-contributing)

</div>

---

ComfyUI has world‑class image and video tooling — but audio has been a patchwork
of one‑off nodes, each dragging in its own conflicting dependencies and often
breaking your install. **ComfyUI-Voice fills that gap**: a single, coherent suite
where every speech model is a drop‑in adapter behind one set of nodes, and
**everything runs natively on the PyTorch you already have** — no second
environment, no dependency hell.

```
text ─▶ [Voice TTS] ─▶ AUDIO ─▶ [Save Audio]            # synthesize
audio ─▶ [Voice ASR] ─▶ TRANSCRIPT + text               # transcribe
ref ──▶ [Voice TTS] ─▶ AUDIO                             # zero‑shot voice clone
```

## ✨ Highlights

- 🧩 **One node per task, many engines.** Pick an engine from a dropdown — the
  node’s inputs adapt to what that engine declares it can do.
- 🛡️ **Runs on *your* torch.** Engines load natively on ComfyUI’s existing
  PyTorch/Transformers. No version pins fighting your install, no separate venvs
  for the supported set. (Conflicting models can opt into isolation — but none of
  the verified models need it.)
- 🌍 **Multilingual & SOTA.** Wraps leading open models for speech synthesis,
  zero‑shot voice cloning, and transcription across dozens of languages.
- 🔌 **Add a model in one file.** A new engine is a single self‑registering
  adapter — no central dispatch, no UI plumbing to touch.
- 🧵 **Composable.** Everything speaks ComfyUI’s native `AUDIO` type, so it chains
  with the built‑in Load/Save/Preview Audio nodes for free.
- ✅ **Honest status.** Models are marked *verified* only after a real
  round‑trip test on this codebase.

## 🎧 Supported engines

> **Verified (✅)** = real inference tested end‑to‑end (synthesize → transcribe
> round‑trip) on the host stack. **Experimental (🧪)** = adapter shipped, deps not
> yet installed/validated — enable via the *Voice Engine Info* node’s install hint.

### Text‑to‑Speech

| Engine (`id`) | Capability | Languages | License | Status |
|---|---|---|---|---|
| **MeloTTS** (`melotts_korean`) | Fast preset TTS | Korean¹ | Apache‑2.0 / MIT | ✅ |
| **CosyVoice 3.0** (`cosyvoice3`) | **Zero‑shot voice clone** + VC | zh · en · ja · ko · de · es · fr · it · ru | Apache‑2.0 | ✅ |
| **Supertonic 3** (`supertonic`) | On‑device preset TTS (ONNX) | 30+ incl. ko | code MIT / model OpenRAIL‑M | ✅ |
| **MMS‑TTS** (`mms_tts_korean`) | Preset TTS (Transformers VITS) | Korean¹ | CC‑BY‑NC ⚠️ | ✅ (eval) |
| **Kokoro‑82M** (`kokoro`) | Ultra‑light preset TTS | en · es · fr · hi · it · pt · ja · zh | Apache‑2.0 | 🧪 |
| **Chatterbox** (`chatterbox`) | Clone + emotion control | 23 langs | MIT | 🧪 |
| **Qwen3‑TTS** (`qwen3_tts`) | Clone + voice design | 10 langs | Apache‑2.0 | 🧪 |
| **OuteTTS 1.0** (`oute_tts`) | Compact LLM‑TTS | 14 langs | Apache‑2.0 | 🧪 |

### Speech‑to‑Text

| Engine (`id`) | Capability | Languages | License | Status |
|---|---|---|---|---|
| **faster‑whisper** (`faster_whisper`) | Fast ASR + word timestamps | 99 langs | MIT | ✅ |
| **Whisper large‑v3** (`whisper_v3`) | ASR (Transformers) | 99 langs | Apache‑2.0 | ✅ |
| **Korean Whisper** (`korean_whisper`) | ASR (ko fine‑tune) | Korean | Apache‑2.0 | 🧪 |
| **Qwen3‑ASR** (`qwen3_asr`) | ASR + forced‑aligner timestamps | 30+ langs | Apache‑2.0 | 🧪 |
| **SenseVoice** (`sensevoice`) | Very fast ASR + emotion/event tags | 5+ langs | custom ⚠️ | 🧪 |
| **WhisperX** (`whisperx`) | ASR + word alignment + diarization | whisper langs | BSD‑2 (+pyannote gated) | 🧪 |

<sub>¹ The shipped checkpoint is Korean; the MeloTTS/MMS families are multilingual — additional language adapters are easy drop‑ins.</sub>

Plus two dependency‑free **reference engines** (`reference_tone`, `reference_asr`)
that let you smoke‑test the whole pipeline on a clean install and serve as the
adapter template.

## 🚀 Quick start

```bash
# 1) Clone into your ComfyUI custom_nodes
cd ComfyUI/custom_nodes
git clone https://github.com/Streamize-llc/ComfyUI-Voice

# 2) Install an engine's deps (example: the workhorses)
pip install faster-whisper          # ASR
pip install librosa g2pkk jamo python-mecab-ko python-mecab-ko-dic num2words anyascii  # MeloTTS frontend
```

Restart ComfyUI, then in the graph:

1. Add **`Voice TTS 🎙️`** → choose an `engine` → type text.
2. Wire its `AUDIO` output into the core **`Save Audio`** (or `Preview Audio`).
3. For STT, add **`Voice ASR (STT) 🎙️`**, feed it any `AUDIO`, read the transcript.

Not sure what’s installed? Drop a **`Voice Engine Info 🎙️`** node — it lists every
engine, its capabilities, and a `pip install …` hint for anything not yet enabled.

> Some heavyweight engines (e.g. CosyVoice 3.0) need a one‑time model/code setup.
> Each adapter’s docstring in `comfyui_voice/engines/` documents its exact steps.

## 🧩 Nodes

| Node | Category | In → Out |
|---|---|---|
| **Voice TTS** | `audio/voice/tts` | text (+ optional `VOICE_REF`) → `AUDIO` |
| **Voice ASR (STT)** | `audio/asr` | `AUDIO` → `VOICE_TRANSCRIPT` + text |
| **Voice Engine Info** | `audio/voice/util` | — → engine/capability report |

Voice cloning is just wiring a reference clip (via core **Load Audio**) into the
TTS node’s `voice_ref` input — no special upload step.

## 🏗️ Architecture

The design borrows ComfyUI core’s own philosophy — *reimplement/port model code to
run on one torch* rather than pip‑installing a stack per model.

- **Composable typed sockets.** Core `AUDIO` (`{"waveform": [B,C,T], "sample_rate"}`)
  plus a small set of namespaced `VOICE_*` types (`VOICE_REF`, `VOICE_TRANSCRIPT`,
  `VOICE_STEMS`, …) make nodes interoperable and future‑proof.
- **Capability‑driven.** Each engine declares one `EngineCapabilities` dataclass
  (languages, cloning, sample rate, isolation, license, param schema). The form,
  validation, and tool surface are **generated** from it — no `if engine == …`
  branching anywhere.
- **Two adapter tiers.**
  1. **Native (`inproc`, preferred):** runs on the host torch — via Transformers
     (the architecture lib already present) or by vendoring the upstream inference
     code with small pure‑python deps + load‑time shims. *All verified models use this.*
  2. **Isolated (`subprocess`):** for engines whose pinned deps genuinely conflict
     — they run in a per‑engine venv behind a uniform worker protocol. Opt‑in via a
     single `isolation="subprocess"` field.
- **Suite‑owned model manager** tracks raw model VRAM (which ComfyUI can’t see) and
  evicts across nodes; a sample‑rate guard prevents silent corruption when chaining.

## ➕ Add an engine in one file

Copy `comfyui_voice/engines/tts/_reference_tone.py` and fill it in:

```python
@register_engine("my_tts")
class MyTTS(BaseEngine):
    CAPS = EngineCapabilities(
        id="my_tts", display_name="My TTS",
        tasks=("tts",), license="Apache-2.0", commercial_safe=True,
        supports_cloning=True, languages=("en", "ko"),
        sample_rate=24000, isolation="inproc",   # or "subprocess" if pins conflict
        pip_install=("my-model",), probe_import=("my_model",),
        param_schema={"params": {...}},
    )
    def load(self):
        import my_model                     # lazy-import heavy deps HERE
        self.m = my_model.load(...)
    def generate(self, task, req):
        wav, sr = self.m.tts(req["text"], ...)
        return {"waveform": wav, "sample_rate": sr}
    def unload(self):
        del self.m
```

That’s the whole contract — it appears in the dropdown automatically, the form
adapts to its `param_schema`, and validation honors its capabilities. No core
files to edit.

> **Before adding deps to the main env, run `pip install --dry-run` and confirm it
> doesn’t move ComfyUI’s `torch` / `transformers` / `numpy`.** If it would, declare
> `isolation="subprocess"`.

## 🗺️ Roadmap

- [x] Core framework, typed sockets, capability registry, isolation runtime
- [x] TTS + ASR nodes; verified native TTS (preset & zero‑shot clone) and ASR
- [ ] More verified engines (Qwen3‑TTS, Chatterbox, …)
- [ ] `VOICE_REF` cloning UX + voice library
- [ ] Voice conversion (RVC / Seed‑VC)
- [ ] Source separation, denoise/enhance, forced alignment & subtitles
- [ ] Music / SFX generation, audio editing

## 🤝 Contributing

PRs and new engine adapters are very welcome. A good contribution:

1. Is a **single adapter file** under `comfyui_voice/engines/<task>/`.
2. Passes the pure‑core tests: `python tests/test_core.py` (no models needed).
3. Keeps the host stack intact (`pip install --dry-run` clean, or `subprocess`).
4. Declares an accurate license + `commercial_safe` flag.

## 📜 License & credits

ComfyUI-Voice is released under the **Apache‑2.0** license. Each wrapped model
keeps its **own** license — see the table above and the per‑engine adapter; some
are non‑commercial or use‑restricted. You are responsible for complying with the
license of any model you enable.

Built on the shoulders of the open‑source speech community — MeloTTS, CosyVoice,
Supertonic, MMS, OpenAI Whisper / faster‑whisper, Kokoro, Chatterbox, Qwen, OuteTTS,
SenseVoice, WhisperX, and ComfyUI itself. Thank you. 🙏
