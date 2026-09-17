"""User-provided checkpoints (e.g. a Spanish fine-tune of F5-TTS) offered as extra engine variants.

Engines are stateless singletons discovered at import time, so the list of custom variants lives in this
process-wide store: the API refreshes it after every change and the lifespan loads it at startup. Nothing here
is invented — a custom variant only declares what the user typed (architecture, checkpoint, vocabulary, languages).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

CUSTOM_PREFIX = "custom:"


def variant_id(slug: str) -> str:
    return f"{CUSTOM_PREFIX}{slug}"


def is_custom(variant: str | None) -> bool:
    return bool(variant and variant.startswith(CUSTOM_PREFIX))


@dataclass(frozen=True)
class CustomVariant:
    """A checkpoint the user added: either a file on disk or a file in a Hugging Face repo."""

    id: str  # "custom:<slug>"
    engine: str
    name: str
    base_variant: str  # architecture config the checkpoint was fine-tuned from (F5TTS_v1_Base…)
    repo_id: str | None = None
    ckpt_file: str | None = None
    local_path: str | None = None
    vocab_file: str | None = None
    vocab_path: str | None = None
    languages: list[str] = field(default_factory=list)
    notes: str | None = None

    @property
    def from_hub(self) -> bool:
        return bool(self.repo_id and self.ckpt_file)

    def resolved_path(self) -> Path | None:
        """Local checkpoint path, if this variant is a file on disk."""
        return Path(self.local_path) if self.local_path else None


class CustomVariantStore:
    def __init__(self) -> None:
        self._by_engine: dict[str, list[CustomVariant]] = {}
        self._lock = Lock()

    def replace(self, variants: list[CustomVariant]) -> None:
        grouped: dict[str, list[CustomVariant]] = {}
        for variant in variants:
            grouped.setdefault(variant.engine, []).append(variant)
        with self._lock:
            self._by_engine = grouped

    def for_engine(self, engine_id: str) -> list[CustomVariant]:
        with self._lock:
            return list(self._by_engine.get(engine_id, ()))

    def get(self, engine_id: str, variant_id_: str) -> CustomVariant | None:
        return next((v for v in self.for_engine(engine_id) if v.id == variant_id_), None)

    def clear(self) -> None:
        with self._lock:
            self._by_engine = {}


_store = CustomVariantStore()


def get_custom_variants() -> CustomVariantStore:
    return _store
