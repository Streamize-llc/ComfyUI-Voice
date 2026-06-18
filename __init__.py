"""ComfyUI-Voice custom-node package.

ComfyUI's loader (nodes.py: load_custom_node) imports this package and calls
``comfy_entrypoint`` (V3 ComfyExtension API). All real code lives in the
``comfyui_voice`` subpackage so it survives ComfyUI upgrades cleanly.
"""
from .comfyui_voice.extension import comfy_entrypoint

__all__ = ["comfy_entrypoint"]
