"""Pure-core tests — NO ComfyUI server, NO model weights, NO comfy_api import.

Run:
    & "C:\\Users\\qkrwn\\anaconda3\\envs\\ComfyUI\\python.exe" \\
        custom_nodes/ComfyUI-Voice/tests/test_core.py

Covers: type-string hygiene, capability serialization, registry + folder scan,
form-spec generation, capability validation, audio shape normalization, and a
full reference-engine TTS run producing a valid AUDIO dict.
"""
from __future__ import annotations

import os
import sys

# Make `comfyui_voice` importable as a top-level package.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import torch  # noqa: E402

from comfyui_voice import audio_utils, types  # noqa: E402
from comfyui_voice.base import BaseEngine, EngineCapabilities  # noqa: E402
from comfyui_voice.processors import asr_processor, gen_processor, tts_processor  # noqa: E402
from comfyui_voice.registry import (  # noqa: E402
    ENGINE_REGISTRY,
    all_capabilities,
    engine_available,
    engines_for_task,
    get_engine,
    scan_engines,
)
from comfyui_voice.runtime import SubprocessRuntime  # noqa: E402
from comfyui_voice.schema_form import build_form_spec, validate_against_caps  # noqa: E402

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


def test_type_hygiene() -> None:
    print("[type hygiene]")
    # Every reserved custom string carries the prefix; AUDIO is the only bare one.
    for s in types.RESERVED_TYPES:
        check(f"{s} has VOICE_ prefix", s.startswith(types.TYPE_PREFIX), s)
    check("AUDIO is reserved", types.is_reserved(types.AUDIO))
    check("unknown type rejected", not types.is_reserved("TRANSCRIPT"))
    # No duplicate strings (dict keys already unique; assert count sanity).
    check("reserved set non-empty", len(types.RESERVED_TYPES) >= 10)


def test_capabilities() -> None:
    print("[capabilities]")
    caps = EngineCapabilities(id="x", display_name="X", tasks=("tts",))
    caps.validate()
    d = caps.to_dict()
    check("to_dict round-trips id", d["id"] == "x")
    check("to_dict is json-able", isinstance(d["dep_pins"], dict))
    try:
        EngineCapabilities(id="", display_name="bad").validate()
        check("empty id rejected", False)
    except ValueError:
        check("empty id rejected", True)


def test_registry_scan() -> None:
    print("[registry scan]")
    scan_engines()
    check("reference_tone registered", "reference_tone" in ENGINE_REGISTRY)
    check("tts task indexed", "reference_tone" in engines_for_task("tts"))
    check("all_capabilities returns caps", "reference_tone" in all_capabilities())


def test_form_spec() -> None:
    print("[form spec]")
    caps = all_capabilities()["reference_tone"]
    fields = build_form_spec(caps)
    names = {f["name"] for f in fields}
    check("form has text/seed/speed", {"text", "seed", "speed"} <= names, str(names))
    text_field = next(f for f in fields if f["name"] == "text")
    check("text field multiline", text_field.get("multiline") is True)


def test_validation() -> None:
    print("[validation]")
    caps = all_capabilities()["reference_tone"]
    check("valid request passes", validate_against_caps(caps, {"text": "hi", "language": "auto"}))
    # An engine that needs ref audio must reject a request without it.
    need_ref = EngineCapabilities(
        id="needs", display_name="N", tasks=("tts",), needs_ref_audio=True
    )
    try:
        validate_against_caps(need_ref, {"text": "hi"})
        check("missing ref audio rejected", False)
    except ValueError:
        check("missing ref audio rejected", True)


def test_audio_utils() -> None:
    print("[audio utils]")
    a1 = audio_utils.to_audio_dict(torch.zeros(100), 24000)
    check("1-D -> [1,1,T]", tuple(a1["waveform"].shape) == (1, 1, 100), str(a1["waveform"].shape))
    a2 = audio_utils.to_audio_dict(torch.zeros(2, 100), 24000)
    check("2-D -> [1,2,T]", tuple(a2["waveform"].shape) == (1, 2, 100))
    a3 = audio_utils.to_audio_dict(torch.zeros(1, 1, 100), 24000)
    check("3-D preserved", tuple(a3["waveform"].shape) == (1, 1, 100))
    r = audio_utils.resample(a1, 16000)
    check("resample changes sr", r["sample_rate"] == 16000)
    check("resample changes length", r["waveform"].shape[-1] != 100)
    check("dtype float32", a1["waveform"].dtype == torch.float32)


def test_end_to_end_tts() -> None:
    print("[end-to-end tts]")
    audio = tts_processor.run_tts(
        "reference_tone", text="안녕하세요. 테스트입니다.", seed=7, params={"speed": 1.0}
    )
    w = audio["waveform"]
    check("returns dict with waveform+sr", "waveform" in audio and "sample_rate" in audio)
    check("waveform is 3-D", w.dim() == 3, str(w.shape))
    check("sample_rate is 24000", audio["sample_rate"] == 24000)
    check("non-empty audio", w.shape[-1] > 1000)
    check("audio in range", float(w.abs().max()) <= 1.0)
    # Determinism: same inputs -> same output.
    audio2 = tts_processor.run_tts("reference_tone", text="안녕하세요. 테스트입니다.", seed=7)
    check("deterministic", torch.equal(audio["waveform"], audio2["waveform"]))
    # Chunking path: very long text should still produce one contiguous clip.
    long_text = "문장. " * 400  # ~2400 chars > max_input_chars(2000)
    long_audio = tts_processor.run_tts("reference_tone", text=long_text, seed=1)
    check("long text chunked+stitched", long_audio["waveform"].shape[-1] > w.shape[-1])


def test_availability() -> None:
    print("[availability]")
    scan_engines()
    # Reference engines declare no probe_import -> always available.
    check("reference_tone available", engine_available("reference_tone"))
    check("reference_asr available", engine_available("reference_asr"))


def test_asr_pipeline() -> None:
    print("[asr pipeline]")
    scan_engines()
    check("reference_asr registered", "reference_asr" in engines_for_task("asr"))
    audio = audio_utils.to_audio_dict(torch.zeros(16000), 16000)  # 1 second
    result = asr_processor.run_asr("reference_asr", audio=audio, language="ko")
    check("asr returns text", bool(result.get("text")))
    check("asr returns segments", len(result.get("segments", [])) >= 1)
    check("asr language passthrough", result.get("language") == "ko")


def test_subprocess_runtime() -> None:
    print("[subprocess runtime]")
    scan_engines()
    caps = get_engine("reference_tone").CAPS
    rt = SubprocessRuntime("reference_tone", caps)  # falls back to host python (no venv)
    try:
        out = rt.generate("tts", {"text": "격리 런타임 테스트", "seed": 7, "speed": 1.0})
        sub = audio_utils.to_audio_dict(out["waveform"], out["sample_rate"])
        inp = tts_processor.run_tts("reference_tone", text="격리 런타임 테스트", seed=7, params={"speed": 1.0})
        check("subprocess produced audio", sub["waveform"].shape[-1] > 1000)
        check("subprocess == inproc (deterministic)",
              torch.allclose(sub["waveform"], inp["waveform"], atol=1e-5),
              f"shapes {tuple(sub['waveform'].shape)} vs {tuple(inp['waveform'].shape)}")
    finally:
        SubprocessRuntime.shutdown_all()


def test_gen_pipeline() -> None:
    print("[gen pipeline (music/sfx)]")
    scan_engines()
    check("reference_music registered", "reference_music" in engines_for_task("music"))
    check("reference_sfx registered", "reference_sfx" in engines_for_task("sfx"))
    music = gen_processor.run_gen("music", "reference_music", text="calm piano", duration=2.0, seed=3)
    sr = music["sample_rate"]
    check("music is 3-D audio", music["waveform"].dim() == 3)
    check("music ~= requested duration", abs(music["waveform"].shape[-1] / sr - 2.0) < 0.1)
    sfx = gen_processor.run_gen("sfx", "reference_sfx", text="thunder", duration=1.5, seed=5)
    check("sfx is 3-D audio", sfx["waveform"].dim() == 3)
    check("sfx ~= requested duration", abs(sfx["waveform"].shape[-1] / sfx["sample_rate"] - 1.5) < 0.1)
    check("gen audio in range", float(music["waveform"].abs().max()) <= 1.0)


def main() -> int:
    for fn in (
        test_type_hygiene,
        test_capabilities,
        test_registry_scan,
        test_form_spec,
        test_validation,
        test_audio_utils,
        test_end_to_end_tts,
        test_availability,
        test_asr_pipeline,
        test_subprocess_runtime,
        test_gen_pipeline,
    ):
        fn()
    print(f"\n{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
