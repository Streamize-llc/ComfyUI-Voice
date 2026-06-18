"""The generic Voice ASR (speech-to-text) node (V3). Engines via a dropdown."""
from __future__ import annotations

from comfy_api.latest import IO

from ..processors import asr_processor
from ..registry import all_capabilities, engines_for_task, scan_engines
from ..types import VOICE_TRANSCRIPT

_NO_ENGINE = "(no ASR engines installed)"


def _engine_options() -> list[str]:
    scan_engines()
    return engines_for_task("asr") or [_NO_ENGINE]


def _language_options() -> list[str]:
    scan_engines()
    langs: set[str] = set()
    for caps in all_capabilities().values():
        if "asr" in caps.tasks:
            langs.update(lang for lang in caps.languages if lang != "any")
    return ["auto", *sorted(langs)]


class VoiceASR(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VoiceASR",
            display_name="Voice ASR (STT) 🎙️",
            category="audio/asr",
            description=(
                "Speech-to-text across a registry of engines. Outputs a "
                "VOICE_TRANSCRIPT (with timestamps) plus the plain text."
            ),
            inputs=[
                IO.Combo.Input("engine", options=_engine_options()),
                IO.Audio.Input("audio"),
                IO.Combo.Input("language", options=_language_options(), default="auto", optional=True),
                IO.Boolean.Input("word_timestamps", default=False, optional=True),
            ],
            outputs=[
                IO.Custom(VOICE_TRANSCRIPT).Output(display_name="transcript"),
                IO.String.Output(display_name="text"),
            ],
        )

    @classmethod
    def execute(cls, engine, audio, language="auto", word_timestamps=False) -> IO.NodeOutput:
        if not engine or engine.startswith("("):
            raise ValueError(
                "No ASR engine installed. Drop an adapter in "
                "comfyui_voice/engines/asr/ (see _reference_echo.py)."
            )
        transcript = asr_processor.run_asr(
            engine,
            audio=audio,
            language=language,
            params={"word_timestamps": bool(word_timestamps)},
        )
        return IO.NodeOutput(transcript, transcript.get("text", ""))
