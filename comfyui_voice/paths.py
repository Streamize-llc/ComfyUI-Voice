"""Filesystem locations for the suite (model cache, per-engine venvs)."""
from __future__ import annotations

import os

# .../ComfyUI-Voice/comfyui_voice/paths.py -> SUITE_ROOT = .../ComfyUI-Voice
SUITE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_DIR = os.path.dirname(os.path.abspath(__file__))  # .../comfyui_voice


def runtimes_dir() -> str:
    return os.path.join(PKG_DIR, "runtimes")


def venv_dir(engine_id: str) -> str:
    return os.path.join(runtimes_dir(), engine_id, ".venv")


def venv_python(engine_id: str) -> str | None:
    """Path to a per-engine venv's python, or None if that venv isn't set up."""
    base = venv_dir(engine_id)
    for candidate in (
        os.path.join(base, "Scripts", "python.exe"),  # Windows
        os.path.join(base, "bin", "python"),  # POSIX
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def models_dir(subdir: str = "") -> str:
    """Model weights live under ComfyUI's models/ tree when available.

    Falls back to a suite-local folder if folder_paths isn't importable
    (e.g. pure-core tests).
    """
    try:
        import folder_paths

        root = os.path.join(folder_paths.models_dir, "voice")
    except Exception:
        root = os.path.join(SUITE_ROOT, "models")
    path = os.path.join(root, subdir) if subdir else root
    return path
