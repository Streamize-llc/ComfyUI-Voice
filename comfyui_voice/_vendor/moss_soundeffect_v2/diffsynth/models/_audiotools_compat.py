"""Minimal stand-in for the two `descript-audiotools` symbols `dac_vae.py` imports.

Upstream MOSS pins ``descript-audiotools==0.7.2``, a heavy package (librosa /
numba / ffmpeg) that would drag numpy and friends off ComfyUI's pinned stack.
The text-to-audio *decode* path only needs:

  * ``BaseModel`` — the nn.Module parent of ``DAC`` whose ``.load()`` classmethod
    reconstructs the model from a ``{"state_dict", "metadata": {"kwargs": ...}}``
    checkpoint (the format of ``vae_128d_48k.pth``), and
  * ``AudioSignal`` — only referenced inside ``compress``/``decompress``/``__main__``,
    which generation never calls.

So we supply a faithful ``BaseModel.load`` and an ``AudioSignal`` placeholder that
raises if it is ever actually used. No third-party install, no env churn.
"""
from __future__ import annotations

import inspect

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Re-implements the slice of ``audiotools.ml.BaseModel`` that DAC relies on."""

    @classmethod
    def load(cls, location, *args, package: bool = True, strict: bool = False, **kwargs):
        # The MOSS DAC checkpoint is a plain ``torch.save`` dict (not a
        # torch.package), so we go straight to the dict path that descript's
        # BaseModel.load falls back to.
        model_dict = torch.load(location, map_location="cpu", weights_only=False)
        metadata = model_dict["metadata"]
        ctor_kwargs = dict(metadata.get("kwargs", {}))
        ctor_kwargs.update(kwargs)
        # Drop any kwarg the constructor doesn't accept (descript does the same).
        valid = set(inspect.signature(cls.__init__).parameters)
        ctor_kwargs = {k: v for k, v in ctor_kwargs.items() if k in valid}
        model = cls(*args, **ctor_kwargs)
        model.load_state_dict(model_dict["state_dict"], strict=strict)
        model.eval()
        return model

    @property
    def device(self):
        for p in self.parameters():
            return p.device
        return torch.device("cpu")


class AudioSignal:  # pragma: no cover - never instantiated on the generate path
    """Placeholder; the generate/decode path does not touch AudioSignal."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "AudioSignal is not available in the ComfyUI-Voice vendored MOSS build "
            "(text-to-audio generation does not require it)."
        )
