"""Audio tensor helpers: normalize to the ComfyUI AUDIO dict + sample-rate guard.

The core AUDIO payload is ``{"waveform": Tensor[batch, channels, samples] float32,
"sample_rate": int}``. Engines may return 1-D / 2-D / 3-D waveforms; we normalize
once here so node code never deals with shape ambiguity. The SR guard prevents
the classic silent corruption when chaining nodes with different native rates.
"""
from __future__ import annotations

import logging

import torch

try:
    import torchaudio

    _HAS_TORCHAUDIO = True
except Exception:  # pragma: no cover
    _HAS_TORCHAUDIO = False

log = logging.getLogger("comfyui_voice.audio")


def to_audio_dict(waveform, sample_rate: int) -> dict:
    """Coerce an arbitrary waveform into a valid AUDIO dict (3-D float32 on CPU)."""
    w = waveform
    if not torch.is_tensor(w):
        w = torch.as_tensor(w, dtype=torch.float32)
    w = w.to(torch.float32)
    if w.dim() == 1:  # [T] -> [1, 1, T]
        w = w.unsqueeze(0).unsqueeze(0)
    elif w.dim() == 2:  # [C, T] -> [1, C, T]
        w = w.unsqueeze(0)
    elif w.dim() != 3:
        raise ValueError(f"waveform must be 1/2/3-D, got {w.dim()}-D shape {tuple(w.shape)}")
    return {"waveform": w.contiguous().cpu(), "sample_rate": int(sample_rate)}


def resample(audio: dict, target_sr: int) -> dict:
    """Resample an AUDIO dict to ``target_sr``."""
    sr = int(audio["sample_rate"])
    if sr == target_sr:
        return audio
    w = audio["waveform"]
    if _HAS_TORCHAUDIO:
        w = torchaudio.functional.resample(w, sr, target_sr)
    else:  # linear-interp fallback so the suite never hard-depends on torchaudio
        b, c, t = w.shape
        new_t = max(1, int(round(t * target_sr / sr)))
        w = torch.nn.functional.interpolate(
            w.reshape(b * c, 1, t), size=new_t, mode="linear", align_corners=False
        ).reshape(b, c, new_t)
    return {"waveform": w, "sample_rate": target_sr}


def ensure_sr(audio: dict, target_sr: int) -> dict:
    """SR guard: resample (with a log) only if the rate differs."""
    if int(audio["sample_rate"]) != int(target_sr):
        log.info("sr_guard: resampling %d -> %d Hz", audio["sample_rate"], target_sr)
        return resample(audio, target_sr)
    return audio


def to_mono_np(audio: dict, target_sr: int = 16000):
    """Downmix an AUDIO dict to mono + resample; return (np.float32[T], target_sr).

    Accepts torch- or numpy-backed waveforms (numpy arrives across the subprocess
    boundary). Used by ASR adapters that expect 16 kHz mono numpy.
    """
    import numpy as np

    w = audio["waveform"]
    sr = int(audio["sample_rate"])
    if not torch.is_tensor(w):
        w = torch.as_tensor(np.asarray(w), dtype=torch.float32)
    w = w.to(torch.float32)
    if w.dim() == 1:
        w = w.view(1, 1, -1)
    elif w.dim() == 2:
        w = w.unsqueeze(0)
    resampled = ensure_sr({"waveform": w, "sample_rate": sr}, target_sr)
    mono = resampled["waveform"].mean(dim=1)[0]  # [B,C,T] -> [T]
    return np.ascontiguousarray(mono.cpu().numpy(), dtype=np.float32), target_sr


def peak_normalize(waveform: torch.Tensor, peak: float = 0.95) -> torch.Tensor:
    m = waveform.abs().max()
    if m > 0:
        waveform = waveform / m * peak
    return waveform


def write_temp_wav(audio: dict) -> str:
    """Write an AUDIO dict to a temp 16-bit WAV (stdlib only) and return its path.

    Used by cloning engines that want a reference-audio file path. Works in any
    venv (no soundfile/torchaudio dependency).
    """
    import os
    import tempfile
    import wave

    import numpy as np

    w = audio["waveform"]
    sr = int(audio["sample_rate"])
    if torch.is_tensor(w):
        w = w.detach().cpu().numpy()
    w = np.asarray(w, dtype=np.float32)
    if w.ndim == 3:  # [B, C, T] -> [C, T]
        w = w[0]
    elif w.ndim == 1:  # [T] -> [1, T]
        w = w[None, :]
    pcm = (np.clip(w, -1.0, 1.0) * 32767.0).astype("<i2").T.tobytes()  # interleaved [T, C]
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="voice_ref_")
    os.close(fd)
    with wave.open(path, "wb") as f:
        f.setnchannels(w.shape[0])
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm)
    return path
