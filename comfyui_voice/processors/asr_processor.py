"""ASR orchestration: validate -> run -> normalize transcript.

Long audio is handled by the engines themselves (faster-whisper etc. chunk
internally), so v0 passes the whole clip. Optional VAD/diarization chaining lands
in later versions via the VOICE_TRANSCRIPT / VOICE_DIARIZATION types.
"""
from __future__ import annotations

import logging

from ..registry import get_engine
from ..runtime import get_runtime

log = logging.getLogger("comfyui_voice.asr")


def run_asr(engine_id: str, *, audio: dict, language: str = "auto", params: dict | None = None) -> dict:
    """Transcribe an AUDIO dict; returns ``{text, segments, language}``."""
    if audio is None or "waveform" not in audio:
        raise ValueError("VoiceASR requires an AUDIO input.")
    caps = get_engine(engine_id).CAPS
    if (
        language
        and language != "auto"
        and caps.languages
        and "any" not in caps.languages
        and language not in caps.languages
    ):
        log.warning("engine '%s' does not declare language '%s'", engine_id, language)

    req = {"audio": audio, "language": language, **(params or {})}
    out = get_runtime(engine_id).generate("asr", req)
    return {
        "text": out.get("text", ""),
        "segments": out.get("segments", []),
        "language": out.get("language", language),
    }
