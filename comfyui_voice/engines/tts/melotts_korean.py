"""MeloTTS (Korean) — NATIVE port, commercial-safe (MIT), runs on host torch 2.12.

This is a true native port (the user's target tier): the MeloTTS model code is
VENDORED under comfyui_voice/_vendor/melo (torch-only VITS) with the all-language
text frontend stripped to Korean-only, so it runs IN-PROCESS on our torch 2.12 /
transformers 5.10 / numpy 2.4 — NO upstream `melotts` pip package (which pins
transformers==4.27.4 + librosa==0.9.1<numpy2), NO subprocess.

Korean G2P: g2pkk + python-mecab-ko (g2pkk is monkeypatched to use python-mecab-ko
on Windows instead of the build-painful `eunjeon`). Prosody BERT: kykim/bert-kor-base
via the host transformers. Verified: real Korean speech @ 44.1kHz on the RTX 4090.
"""
from __future__ import annotations

import os
from typing import Any

from ...base import BaseEngine, EngineCapabilities
from ...paths import PKG_DIR
from ...registry import register_engine

_VENDOR = os.path.join(PKG_DIR, "_vendor")
_patched = False


def _ensure_melo_importable() -> None:
    """Put the vendored melo on sys.path and make g2pkk use python-mecab-ko."""
    global _patched
    import sys

    if _VENDOR not in sys.path:
        sys.path.insert(0, _VENDOR)
    if not _patched:
        import g2pkk.g2pkk as _g  # g2pkk is a small pure dep

        _g.G2p.check_mecab = lambda self: None  # skip the Windows eunjeon auto-install
        _g.G2p.get_mecab = lambda self: __import__("mecab").MeCab()  # python-mecab-ko
        _patched = True


@register_engine("melotts_korean")
class MeloTTSKorean(BaseEngine):
    CAPS = EngineCapabilities(
        id="melotts_korean",
        display_name="MeloTTS Korean (native · torch 2.12)",
        tasks=("tts",),
        license="MIT",
        commercial_safe=True,
        supports_cloning=False,
        languages=("ko",),
        sample_rate=44100,
        vram_est_gb=1.5,
        isolation="inproc",  # native vendored port — no isolation needed
        max_input_chars=2000,
        pip_install=(
            "librosa", "g2pkk", "jamo", "python-mecab-ko", "python-mecab-ko-dic",
            "num2words", "anyascii", "unidecode", "cached_path",
        ),
        probe_import=("g2pkk", "mecab", "librosa", "jamo", "num2words", "anyascii"),
        param_schema={
            "params": {
                "text": {"type": "string", "multiline": True, "default": "안녕하세요. 멜로 티티에스입니다.", "targets": ["text"]},
                "speed": {"type": "float", "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "targets": ["speed"]},
            }
        },
    )

    def load(self) -> None:
        _ensure_melo_importable()
        from melo.api import TTS

        self._tts = TTS(language="KR", device="auto")
        self._spk = self._tts.hps.data.spk2id["KR"]
        self._sr = int(self._tts.hps.data.sampling_rate)

    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        text = str(req.get("text") or "").strip()
        speed = float(req.get("speed", 1.0)) or 1.0
        wav = self._tts.tts_to_file(text, self._spk, output_path=None, speed=speed, quiet=True)
        return {"waveform": np.asarray(wav, dtype=np.float32).reshape(-1), "sample_rate": self._sr}
