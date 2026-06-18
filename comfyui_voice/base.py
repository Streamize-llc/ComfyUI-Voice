"""Engine adapter base class + the single source of truth: EngineCapabilities.

The whole suite is driven by one declarative object per engine. A contributor
adds a model by subclassing :class:`BaseEngine`, declaring a
:class:`EngineCapabilities` as the ``CAPS`` class attribute, and decorating with
``@register_engine``. The UI form, the request validator, the (future) MCP tool
schema and the docs table are all GENERATED from ``CAPS`` — never branch on the
engine id anywhere else.
"""
from __future__ import annotations

import abc
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Literal


@dataclass
class EngineCapabilities:
    """Everything the suite needs to know about an engine, declaratively.

    Serializable to JSON (``to_dict``) so it can drive the web form, validation,
    MCP tools and the docs comparison table from a single place.
    """

    # --- identity ---
    id: str
    display_name: str
    tasks: tuple[str, ...] = ("tts",)  # ("tts", "voice_conversion", ...) -> routing
    version: str = "0"

    # --- license / compliance (a GATE, not a badge) ---
    license: str = "unknown"
    commercial_safe: bool = False
    weights_gated: bool = False  # needs HF acceptance / login

    # --- cloning ---
    supports_cloning: bool = False
    needs_ref_audio: bool = False
    needs_ref_text: bool = False
    max_ref_audio_sec: float | None = None

    # --- language / style ---
    languages: tuple[str, ...] = ()  # BCP-47 codes; () = unknown; ("any",) = any
    supports_emotion: bool = False
    emotion_modes: tuple[str, ...] = ()

    # --- length / streaming ---
    supports_streaming: bool = False
    supports_long_form: bool = False
    max_input_chars: int | None = None

    # --- output / resources ---
    sample_rate: int = 24000
    output_channels: int = 1
    vram_est_gb: float | None = None
    model_size_gb: float | None = None

    # --- dependency isolation (a declared field, not a rewrite) ---
    isolation: Literal["inproc", "subprocess"] = "inproc"
    dep_pins: dict[str, str] = field(default_factory=dict)
    # How to install this engine's deps + how to detect if they're present.
    # The adapter MODULE stays importable without these (CAPS is dep-free);
    # only load() needs them. probe_import is checked to report availability.
    pip_install: tuple[str, ...] = ()
    probe_import: tuple[str, ...] = ()

    # --- schema: drives form / validation / MCP (same shape as workflows.py) ---
    param_schema: dict[str, Any] = field(default_factory=dict)

    # --- runtime quirks ---
    requires_special_init: bool = False
    can_corrupt_on_reload: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if not self.id:
            raise ValueError("EngineCapabilities.id must be non-empty")
        if not self.tasks:
            raise ValueError(f"engine '{self.id}': tasks must be non-empty")
        if self.sample_rate <= 0:
            raise ValueError(f"engine '{self.id}': sample_rate must be > 0")
        if self.isolation not in ("inproc", "subprocess"):
            raise ValueError(f"engine '{self.id}': isolation must be inproc|subprocess")


class BaseEngine(abc.ABC):
    """Adapter contract. Keep it small and raw.

    Chunking, streaming, SRT timing and sample-rate normalization live in the
    shared processors — an adapter only does load / generate-once / unload.
    """

    CAPS: ClassVar[EngineCapabilities]

    def __init__(self) -> None:
        self._loaded = False

    def load(self) -> None:
        """Load weights into memory. Lazy-import heavy deps INSIDE this method."""
        self._loaded = True

    @abc.abstractmethod
    def generate(self, task: str, req: dict[str, Any]) -> dict[str, Any]:
        """Run one generation.

        ``req`` carries task-specific keys (e.g. text/ref_audio for tts).
        Returns a payload dict — for tts: ``{"waveform": Tensor, "sample_rate": int}``.
        """

    def unload(self) -> None:
        self._loaded = False

    @classmethod
    def capabilities(cls) -> EngineCapabilities:
        return cls.CAPS
