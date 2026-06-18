"""Generative-audio nodes: text -> music, text -> sound effects.

One generic node per task, engines via the registry dropdown — same pattern as
VoiceTTS/VoiceASR. Both output the core AUDIO type.
"""
from __future__ import annotations

from comfy_api.latest import IO

from ..processors import gen_processor
from ..registry import engines_for_task, scan_engines


def _options(task: str, label: str) -> list[str]:
    scan_engines()
    return engines_for_task(task) or [f"(no {label} engines installed)"]


class VoiceMusicGen(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VoiceMusicGen",
            display_name="Voice Music Gen 🎙️",
            category="audio/generate/music",
            description="Text-to-music generation across a registry of engines.",
            inputs=[
                IO.Combo.Input("engine", options=_options("music", "music")),
                IO.String.Input("text", multiline=True, default="calm lofi piano, 90 bpm"),
                IO.Float.Input("duration", default=10.0, min=1.0, max=240.0, step=1.0, optional=True),
                IO.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True, optional=True),
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    def execute(cls, engine, text, duration=10.0, seed=42) -> IO.NodeOutput:
        if not engine or engine.startswith("("):
            raise ValueError("No music engine installed. Drop an adapter in comfyui_voice/engines/music/.")
        audio = gen_processor.run_gen("music", engine, text=text, duration=duration, seed=seed)
        return IO.NodeOutput(audio)


class VoiceSFXGen(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VoiceSFXGen",
            display_name="Voice SFX Gen 🎙️",
            category="audio/generate/sfx",
            description="Text-to-sound-effects / Foley generation across a registry of engines.",
            inputs=[
                IO.Combo.Input("engine", options=_options("sfx", "SFX")),
                IO.String.Input("text", multiline=True, default="rain on a window, distant thunder"),
                IO.Float.Input("duration", default=5.0, min=0.5, max=60.0, step=0.5, optional=True),
                IO.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True, optional=True),
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    def execute(cls, engine, text, duration=5.0, seed=42) -> IO.NodeOutput:
        if not engine or engine.startswith("("):
            raise ValueError("No SFX engine installed. Drop an adapter in comfyui_voice/engines/sfx/.")
        audio = gen_processor.run_gen("sfx", engine, text=text, duration=duration, seed=seed)
        return IO.NodeOutput(audio)
