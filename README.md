# ComfyUI-Voice 🎙️

An **extensible, open-source audio/voice node suite** for ComfyUI. The goal is to
own the (currently empty) space of a *proper, unified* audio toolkit — TTS,
voice cloning, voice conversion, ASR, source separation, enhancement, music/SFX
generation and more — under one coherent, contributor-friendly architecture.

> **One mantra:** *one generic node per task + single-file engine adapter +
> capability declaration.* Adding a model = drop one file in
> `comfyui_voice/engines/<task>/`. Nothing else changes.

## Status — v0

- ✅ Core framework: typed sockets, engine registry, model manager, runtime
  routing (in-proc / subprocess), SR guard, capability-driven form + validation.
- ✅ Generic `Voice TTS 🎙️` node + `Voice Engine Info 🎙️` node (V3 IO API).
- ✅ A dependency-free **reference engine** (`reference_tone`) so the whole
  pipeline is verifiable on a clean install and serves as the adapter template.
- ⏭️ Next: real TTS engines (MeloTTS / Chatterbox / CosyVoice), `VOICE_REF`
  cloning, voice conversion, then ASR / separation / enhance.

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
