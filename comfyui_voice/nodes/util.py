"""Utility nodes. v0: a registry inspector (handy for debugging / E2E checks)."""
from __future__ import annotations

import json

from comfy_api.latest import IO

from ..registry import all_capabilities, engine_available, install_hint, scan_engines

_TASKS = ["all", "tts", "voice_conversion", "asr", "music", "sfx", "separate", "enhance"]


class VoiceEngineInfo(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VoiceEngineInfo",
            display_name="Voice Engine Info 🎙️",
            category="audio/voice/util",
            description="Lists installed voice engines and their declared capabilities.",
            inputs=[
                IO.Combo.Input("filter_task", options=_TASKS, default="all", optional=True),
            ],
            outputs=[IO.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, filter_task="all") -> IO.NodeOutput:
        scan_engines()
        rows = []
        for engine_id, caps in sorted(all_capabilities().items()):
            if filter_task != "all" and filter_task not in caps.tasks:
                continue
            available = engine_available(engine_id)
            rows.append(
                {
                    "id": engine_id,
                    "display_name": caps.display_name,
                    "tasks": list(caps.tasks),
                    "license": caps.license,
                    "commercial_safe": caps.commercial_safe,
                    "languages": list(caps.languages),
                    "sample_rate": caps.sample_rate,
                    "cloning": caps.supports_cloning,
                    "isolation": caps.isolation,
                    "available": available,
                    **({} if available else {"enable": install_hint(engine_id)}),
                }
            )
        report = json.dumps(rows, ensure_ascii=False, indent=2) if rows else "(no engines installed)"
        return IO.NodeOutput(report)
