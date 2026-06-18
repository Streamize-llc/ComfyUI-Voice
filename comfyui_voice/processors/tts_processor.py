"""TTS orchestration: validate -> chunk -> generate-per-chunk -> stitch.

Chunking/long-form/stitching live HERE (shared), not in adapters — an adapter
only does generate-once. This is where streaming and SRT timing will also land.
"""
from __future__ import annotations

import logging
import re

import torch

from ..audio_utils import ensure_sr, to_audio_dict
from ..registry import get_engine
from ..runtime import get_runtime
from ..schema_form import validate_against_caps

log = logging.getLogger("comfyui_voice.tts")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？\n])\s+")


def _chunk_text(text: str, max_chars: int | None) -> list[str]:
    if not max_chars or len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT.split(text):
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip() if current else sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text]


def run_tts(
    engine_id: str,
    *,
    text: str,
    ref_audio: dict | None = None,
    ref_text: str | None = None,
    language: str = "auto",
    seed: int = 42,
    params: dict | None = None,
    target_sr: int | None = None,
) -> dict:
    """Synthesize ``text`` with ``engine_id`` and return a normalized AUDIO dict."""
    params = dict(params or {})
    caps = get_engine(engine_id).CAPS

    req = {
        "text": text,
        "ref_audio": ref_audio,
        "ref_text": ref_text,
        "language": language,
        "seed": seed,
        **params,
    }
    validate_against_caps(caps, req)
    runtime = get_runtime(engine_id)

    if caps.supports_long_form or not caps.max_input_chars:
        chunks = [text]
    else:
        chunks = _chunk_text(text, caps.max_input_chars)
    if len(chunks) > 1:
        log.info("engine '%s': chunked into %d segments", engine_id, len(chunks))

    waveforms: list[torch.Tensor] = []
    sr = caps.sample_rate
    for chunk in chunks:
        chunk_req = dict(req)
        chunk_req["text"] = chunk
        out = runtime.generate("tts", chunk_req)
        audio = to_audio_dict(out["waveform"], out.get("sample_rate", sr))
        sr = audio["sample_rate"]
        waveforms.append(audio["waveform"])

    waveform = waveforms[0] if len(waveforms) == 1 else torch.cat(waveforms, dim=-1)
    audio = {"waveform": waveform, "sample_rate": sr}
    if target_sr:
        audio = ensure_sr(audio, target_sr)
    return audio
