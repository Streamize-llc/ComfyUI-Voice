"""Turn an EngineCapabilities into UI fields + validate requests against it.

This is the audio-suite analogue of ``comfyui_api/workflows.py``'s
``build_form_spec`` / ``apply_params``: one declaration drives the web form, the
request validator and (later) the MCP tool schema. Capability flags gate inputs
so the UI/API adapt per engine instead of hardcoding model-specific branches.
"""
from __future__ import annotations

from typing import Any

from .base import EngineCapabilities


def build_form_spec(caps: EngineCapabilities, live_options: dict[str, list] | None = None) -> list[dict]:
    """Ordered list of form fields derived from ``caps.param_schema`` + live options.

    ``live_options`` injects current dynamic enum values (e.g. installed voices)
    keyed by a param's ``options_source``.
    """
    live_options = live_options or {}
    fields: list[dict] = []
    params = (caps.param_schema or {}).get("params", {})
    for name, spec in params.items():
        field: dict[str, Any] = {
            "name": name,
            "type": spec.get("type", "string"),
            "default": spec.get("default"),
        }
        for key in ("min", "max", "step", "multiline", "options"):
            if key in spec:
                field[key] = spec[key]
        source = spec.get("options_source")
        if source and source in live_options:
            field["options"] = live_options[source]
        fields.append(field)
    return fields


def validate_against_caps(caps: EngineCapabilities, req: dict[str, Any]) -> bool:
    """Reject requests an engine cannot satisfy. Raises ValueError on hard errors."""
    errors: list[str] = []

    if caps.needs_ref_audio and not req.get("ref_audio"):
        errors.append(f"engine '{caps.id}' requires a reference audio (needs_ref_audio).")

    if caps.needs_ref_text and not str(req.get("ref_text") or "").strip():
        errors.append(f"engine '{caps.id}' requires reference text (needs_ref_text).")

    if req.get("ref_audio") and not caps.supports_cloning:
        # not fatal — just ignore the ref; warn via the message channel later
        pass

    lang = req.get("language")
    if (
        lang
        and lang != "auto"
        and caps.languages
        and "any" not in caps.languages
        and lang not in caps.languages
    ):
        errors.append(
            f"engine '{caps.id}' does not declare language '{lang}' "
            f"(declares {list(caps.languages)})."
        )

    if errors:
        raise ValueError(" ".join(errors))
    return True
