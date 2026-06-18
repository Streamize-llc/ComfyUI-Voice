"""Reserved custom socket-type strings for the ComfyUI-Voice suite.

ComfyUI has NO type registry — socket compatibility is pure string-set
intersection (see comfy_execution/validation.py). So two suites that both pick
``TRANSCRIPT`` for incompatible payloads will validate green and then crash deep
in ``execute()``. Our defense:

1. Every suite-private type carries the ``VOICE_`` prefix. Only the bare core
   ``AUDIO`` string is reused unprefixed (interop with core LoadAudio/SaveAudio
   is the whole point there).
2. Every custom string is declared HERE, in ``RESERVED_TYPES``, and a CI lint
   asserts no adapter emits an undeclared string.

**These strings are a public API contract: append-only and immutable forever.**
Never rename one — it silently breaks every saved workflow that uses it.
Deprecate via alias instead.
"""
from __future__ import annotations

from typing import Any, TypedDict

try:  # torch is always present in a ComfyUI env; keep import soft for pure tooling
    import torch

    _Tensor = torch.Tensor
except Exception:  # pragma: no cover
    _Tensor = Any  # type: ignore

# --- prefix -----------------------------------------------------------------
TYPE_PREFIX = "VOICE_"

# --- core type (reused as-is, NO prefix) ------------------------------------
AUDIO = "AUDIO"

# --- suite-private types (frozen, append-only) ------------------------------
VOICE_REF = "VOICE_REF"
VOICE_MODEL = "VOICE_MODEL"
VOICE_TRANSCRIPT = "VOICE_TRANSCRIPT"
VOICE_ALIGNMENT = "VOICE_ALIGNMENT"
VOICE_SUBTITLE = "VOICE_SUBTITLE"
VOICE_DIARIZATION = "VOICE_DIARIZATION"
VOICE_STEMS = "VOICE_STEMS"
VOICE_SPEAKER_EMBEDDING = "VOICE_SPEAKER_EMBEDDING"
VOICE_AUDIO_TOKENS = "VOICE_AUDIO_TOKENS"
VOICE_STYLE = "VOICE_STYLE"
VOICE_MASK = "VOICE_MASK"
VOICE_WATERMARK = "VOICE_WATERMARK"

# Maps every reserved custom string -> one-line payload doc. The CI lint and
# GET /engines docs read this. Add a row BEFORE using a new type anywhere.
RESERVED_TYPES: dict[str, str] = {
    VOICE_REF: "{audio: AudioDict, ref_text: str|None, encoding: Tensor|None, consent: bool}",
    VOICE_MODEL: "{kind: str, paths: dict, meta: dict} — a trained/savable voice asset",
    VOICE_TRANSCRIPT: "{text: str, segments: [{start,end,text,conf}], language: str}",
    VOICE_ALIGNMENT: "{tokens: [{token,start,end,pitch?}], unit: 'word'|'phoneme'}",
    VOICE_SUBTITLE: "{format: 'srt'|'vtt', cues: [{idx,start,end,text,style?}]}",
    VOICE_DIARIZATION: "{turns: [{speaker_id,start,end,embedding?}]}",
    VOICE_STEMS: "{stems: {name: AudioDict}, sample_rate: int}",
    VOICE_SPEAKER_EMBEDDING: "{vector: Tensor[D], model: str}",
    VOICE_AUDIO_TOKENS: "{codes: Tensor[B,Q,T], codec: str, frame_rate: float}",
    VOICE_STYLE: "{emotion: str|None, intensity: float, rate: float, pitch: float, style_ref: AudioDict|None}",
    VOICE_MASK: "{spans: [{start,end}], sample_rate: int}",
    VOICE_WATERMARK: "{payload: bytes|None}",
}


# --- payload TypedDicts (documentation + editor help) ------------------------
class AudioDict(TypedDict):
    """The core ComfyUI AUDIO payload. 3-D waveform [batch, channels, samples]."""

    waveform: _Tensor
    sample_rate: int


class VoiceRefDict(TypedDict, total=False):
    audio: AudioDict
    ref_text: str | None
    encoding: Any
    consent: bool


class TranscriptSegment(TypedDict, total=False):
    start: float
    end: float
    text: str
    conf: float


class TranscriptDict(TypedDict, total=False):
    text: str
    segments: list[TranscriptSegment]
    language: str


def is_reserved(type_string: str) -> bool:
    """True if ``type_string`` is the core AUDIO type or a declared custom type."""
    return type_string == AUDIO or type_string in RESERVED_TYPES
