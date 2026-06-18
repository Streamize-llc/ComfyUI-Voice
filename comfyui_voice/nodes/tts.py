"""The generic Voice TTS node (V3). One node, engines via a dropdown.

Reads the engine registry to populate the ``engine`` dropdown and the union of
declared languages. Per-engine input adaptation (emotion sliders, ref-text, ...)
is generated from each engine's ``param_schema`` + capabilities — there is no
model-specific branching in this node.
"""
from __future__ import annotations

from comfy_api.latest import IO

from ..processors import tts_processor
from ..registry import all_capabilities, engines_for_task, scan_engines
from ..types import VOICE_REF

_NO_ENGINE = "(no TTS engines installed)"


def _engine_options() -> list[str]:
    scan_engines()
    return engines_for_task("tts") or [_NO_ENGINE]


def _language_options() -> list[str]:
    scan_engines()
    langs: set[str] = set()
    for caps in all_capabilities().values():
        if "tts" in caps.tasks:
            langs.update(lang for lang in caps.languages if lang != "any")
    return ["auto", *sorted(langs)]


class VoiceTTS(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VoiceTTS",
            display_name="Voice TTS 🎙️",
            category="audio/voice/tts",
            description=(
                "Text-to-speech across a registry of engines. Pick an engine; "
                "outputs a standard AUDIO you can feed to Save Audio / Preview Audio."
            ),
            inputs=[
                IO.Combo.Input("engine", options=_engine_options()),
                IO.String.Input(
                    "text",
                    multiline=True,
                    default="안녕하세요. ComfyUI Voice 입니다.",
                ),
                IO.Combo.Input(
                    "language", options=_language_options(), default="auto", optional=True
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    optional=True,
                ),
                IO.Float.Input(
                    "speed", default=1.0, min=0.5, max=2.0, step=0.05, optional=True
                ),
                IO.Custom(VOICE_REF).Input("voice_ref", optional=True),
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    def execute(
        cls,
        engine,
        text,
        language="auto",
        seed=42,
        speed=1.0,
        voice_ref=None,
    ) -> IO.NodeOutput:
        if not engine or engine.startswith("("):
            raise ValueError(
                "No TTS engine installed. Drop an adapter in "
                "comfyui_voice/engines/tts/ (see _reference_tone.py)."
            )
        ref_audio = ref_text = None
        if voice_ref:
            ref_audio = voice_ref.get("audio")
            ref_text = voice_ref.get("ref_text")
        audio = tts_processor.run_tts(
            engine,
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            language=language,
            seed=seed,
            params={"speed": speed},
        )
        return IO.NodeOutput(audio)
