"""Shared model-weight download helper.

Engines call ``ensure_hf_model`` inside ``load()`` so weights land under one
shared tree (ComfyUI ``models/voice/...``) instead of each adapter caching to its
own HF_HOME and duplicating multi-GB files.
"""
from __future__ import annotations

import logging
import os

from .paths import models_dir

log = logging.getLogger("comfyui_voice.downloads")


def ensure_hf_model(
    repo_id: str,
    subdir: str | None = None,
    allow_patterns: list[str] | None = None,
) -> str:
    """Download a HuggingFace repo into models/voice/<subdir> and return the path."""
    target = models_dir(subdir or repo_id.replace("/", "__"))
    os.makedirs(target, exist_ok=True)
    from huggingface_hub import snapshot_download

    log.info("ensuring weights %s -> %s", repo_id, target)
    return snapshot_download(repo_id=repo_id, local_dir=target, allow_patterns=allow_patterns)
