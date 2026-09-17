"""TTS engine plugin contract.

Every engine describes itself declaratively (capabilities + parameter schema). The frontend renders
controls from that description, so adding an engine never requires UI changes. Heavy dependencies
(torch, model packages) must only be imported inside `load()` / `generate()`.
"""

from __future__ import annotations

import importlib.metadata
import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from app.core.errors import AppError, ErrorCode

SEED_MAX = 2**31 - 1


# ---------------------------------------------------------------------------
# Descriptive models (serialised to the frontend)
# ---------------------------------------------------------------------------
class ControlSource(StrEnum):
    NATIVE = "native"              # real model parameter
    INSTRUCTION = "instruction"    # natural-language instruction understood by the model
    SEGMENTATION = "segmentation"  # achieved by splitting text / switching references
    DSP = "dsp"                    # optional post-processing, separate from the model
    UNAVAILABLE = "unavailable"


class GenericControl(StrEnum):
    """Engine-independent UI controls. Each engine states how (or whether) it supports them."""

    SPEED = "speed"
    PITCH = "pitch"
    EMOTION = "emotion"
    NATURALNESS = "naturalness"
    VOICE_COLOR = "voice_color"
    INSTRUCTION = "instruction"
    SEED = "seed"
    TARGET_DURATION = "target_duration"
    LANGUAGE = "language"


class ControlCapability(BaseModel):
    source: ControlSource
    parameter: str | None = Field(default=None, description="ParameterSpec.id que implementa el control")
    reason: str | None = Field(default=None, description="Explicación en español (obligatoria si no es nativo)")
    available_from_phase: int | None = Field(default=None, description="Fase en la que se implementa, si aún no")


class LicenseInfo(BaseModel):
    code: str
    weights: str
    commercial_use: Literal["permitido", "no_permitido", "revisar"]
    notes: str | None = None
    url: str


class EngineVariant(BaseModel):
    id: str
    label: str
    description: str
    mode: Literal["clone", "custom_voice", "voice_design"] = "clone"
    repo_id: str | None = None
    vram_estimate_mb: int | None = Field(default=None, description="Estimación, no medición")
    download_size_mb: int | None = None


ReferenceStrategy = Literal["best_reference", "profile", "per_segment"]


class EngineCapabilities(BaseModel):
    variant: str
    mode: Literal["clone", "custom_voice", "voice_design"]
    sample_rate: int
    languages: list[str]
    requires_reference_audio: bool
    requires_reference_text: bool
    reference_text_optional_reason: str | None = None
    reference_duration_s: tuple[float, float] | None = None
    reference_strategies: list[ReferenceStrategy]
    controls: dict[GenericControl, ControlCapability]
    supports_streaming: bool = False
    supports_batch: bool = False
    deterministic_seed: bool = False
    reports_progress: bool = Field(default=True, description="El motor informa del progreso durante la generación")
    max_chars_per_call: int | None = Field(
        default=None, description="Texto máximo por llamada al modelo; los textos más largos se dividen por frases")
    reference_text_not_required_when: dict[str, Any] | None = Field(
        default=None, description="Valores de parámetros con los que la transcripción de la referencia es opcional")
    notes: list[str] = []


class ParameterTooltip(BaseModel):
    what: str
    effect: str
    cost: str
    typical: str


class SelectOption(BaseModel):
    value: str | int | float | bool
    label: str
    description: str | None = None


ParamType = Literal["slider", "number", "toggle", "select", "text", "seed"]
PresetName = Literal["fast", "balanced", "high_fidelity"]


class ParameterSpec(BaseModel):
    id: str
    maps_to: str | None = Field(default=None, description="Nombre real del argumento en el modelo")
    type: ParamType
    value_type: Literal["int", "float", "bool", "str"]
    label: str
    tooltip: ParameterTooltip
    default: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    unit: str | None = None
    nullable: bool = False
    options: list[SelectOption] | None = None
    dynamic_options: bool = Field(default=False, description="Las opciones reales se obtienen del modelo cargado")
    max_length: int | None = None
    group: Literal["basic", "advanced"] = "advanced"
    control: GenericControl | None = Field(default=None, description="Control genérico que representa")
    visible_if: dict[str, Any] | None = None
    presets: dict[PresetName, Any] = {}


class ResourceEstimate(BaseModel):
    vram_mb: int | None
    download_size_mb: int | None
    is_estimate: bool = True


# ---------------------------------------------------------------------------
# Runtime models (used from FASE 5)
# ---------------------------------------------------------------------------
class ReferenceInput(BaseModel):
    audio_path: Path
    text: str | None = None
    sha256: str
    start_s: float | None = None
    end_s: float | None = None


class PreparedReference(BaseModel):
    engine: str
    variant: str
    cache_key: str
    payload: dict[str, Any] = {}


class EngineRequest(BaseModel):
    variant: str
    text: str
    params: dict[str, Any]
    reference: PreparedReference | None = None


class EngineResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    audio: np.ndarray
    sample_rate: int
    seed: int | None
    effective_params: dict[str, Any]
    warnings: list[str] = []
    details: dict[str, Any] = {}  # runtime facts worth storing in metrics (dtype, attention, …)


ProgressCallback = Callable[[float, str | None], None]


class CancelToken:
    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
PRESET_LABELS: dict[str, str] = {"fast": "Rápido", "balanced": "Equilibrado", "high_fidelity": "Alta fidelidad"}


class TTSBackend(ABC):
    id: str
    display_name: str
    description: str
    license: LicenseInfo
    #: pip distributions that must be installed for the engine to run
    required_packages: tuple[str, ...] = ()
    implementation_phase: int | None = None  # set while generate() is not implemented yet

    # ----- description -----
    @abstractmethod
    def variants(self) -> list[EngineVariant]: ...

    def default_variant(self) -> str:
        return self.variants()[0].id

    def variant(self, variant_id: str | None) -> EngineVariant:
        variant_id = variant_id or self.default_variant()
        for v in self.variants():
            if v.id == variant_id:
                return v
        raise AppError(ErrorCode.ENGINE_NOT_FOUND, status_code=404,
                       message="La variante de modelo solicitada no existe.",
                       details={"motor": self.id, "variante": variant_id})

    @abstractmethod
    def capabilities(self, variant: str | None = None) -> EngineCapabilities: ...

    @abstractmethod
    def get_parameters_schema(self, variant: str | None = None) -> list[ParameterSpec]: ...

    def requirements(self, variant: str | None = None) -> ResourceEstimate:
        v = self.variant(variant)
        return ResourceEstimate(vram_mb=v.vram_estimate_mb, download_size_mb=v.download_size_mb)

    def missing_packages(self) -> list[str]:
        missing = []
        for dist in self.required_packages:
            try:
                importlib.metadata.version(dist)
            except importlib.metadata.PackageNotFoundError:
                missing.append(dist)
        return missing

    def is_installed(self) -> bool:
        return not self.missing_packages()

    # ----- parameters -----
    def resolve_preset(self, preset: PresetName, variant: str | None = None) -> dict[str, Any]:
        return {spec.id: spec.presets.get(preset, spec.default) for spec in self.get_parameters_schema(variant)}

    def requires_reference_text(self, params: dict[str, Any], variant: str | None = None) -> bool:
        caps = self.capabilities(variant)
        exempt = caps.reference_text_not_required_when
        if exempt and all(params.get(k) == v for k, v in exempt.items()):
            return False
        return caps.requires_reference_text

    def validate_parameters(self, params: dict[str, Any], variant: str | None = None) -> dict[str, Any]:
        """Coerce and check values against the schema. Unknown ids are rejected, missing ones get defaults."""
        schema = {s.id: s for s in self.get_parameters_schema(variant)}
        errors: list[dict[str, Any]] = []
        unknown = sorted(set(params) - set(schema))
        for pid in unknown:
            errors.append({"parametro": pid, "error": "No existe para este modelo."})

        result: dict[str, Any] = {}
        for pid, spec in schema.items():
            raw = params.get(pid, spec.default)
            try:
                result[pid] = _coerce(spec, raw)
            except ValueError as exc:
                errors.append({"parametro": pid, "etiqueta": spec.label, "error": str(exc)})
        if errors:
            raise AppError(ErrorCode.PARAMETER_ERROR, status_code=422, details=errors)
        return result

    # ----- weights -----
    def weights_installed(self, variant: str) -> bool:
        """Whether the checkpoint files for `variant` are available locally (no network)."""
        return True

    def download_weights(self, variant: str) -> None:
        """Download checkpoint files for `variant`. Only called after an explicit user action."""
        return None

    # ----- lifecycle (FASE 5+) -----
    def _not_implemented(self) -> AppError:
        return AppError(ErrorCode.NOT_IMPLEMENTED, status_code=501,
                        message=f"La generación con {self.display_name} estará disponible en la fase "
                                f"{self.implementation_phase}.",
                        details={"motor": self.id})

    def load(self, variant: str, device: str, dtype: str) -> None:
        raise self._not_implemented()

    def unload(self) -> None:
        return None

    @property
    def loaded_variant(self) -> str | None:
        return None

    def prepare_reference(self, reference: ReferenceInput, variant: str) -> PreparedReference:
        raise self._not_implemented()

    def generate(self, request: EngineRequest, progress: ProgressCallback, cancel: CancelToken) -> EngineResult:
        raise self._not_implemented()


def _coerce(spec: ParameterSpec, raw: Any) -> Any:
    if raw is None or raw == "":
        if spec.type == "seed":
            return None  # random seed chosen at generation time
        if spec.nullable:
            return None
        raise ValueError("Valor obligatorio.")

    if spec.type == "seed":
        value = int(raw) if not isinstance(raw, bool) else _fail("Debe ser un número entero.")
        if not 0 <= value <= SEED_MAX:
            raise ValueError(f"Debe estar entre 0 y {SEED_MAX}.")
        return value

    if spec.value_type == "bool":
        if isinstance(raw, bool):
            return raw
        if raw in ("true", "false"):
            return raw == "true"
        raise ValueError("Debe ser verdadero o falso.")

    if spec.value_type in ("int", "float"):
        if isinstance(raw, bool):
            raise ValueError("Debe ser un número.")
        try:
            number = float(raw)
        except (TypeError, ValueError):
            raise ValueError("Debe ser un número.") from None
        if not np.isfinite(number):
            raise ValueError("Debe ser un número finito.")
        if spec.value_type == "int":
            if number != int(number):
                raise ValueError("Debe ser un número entero.")
            number = int(number)
        if spec.min is not None and number < spec.min:
            raise ValueError(f"El mínimo es {spec.min:g}.")
        if spec.max is not None and number > spec.max:
            raise ValueError(f"El máximo es {spec.max:g}.")
        value: Any = number
    else:
        value = str(raw)
        if spec.max_length is not None and len(value) > spec.max_length:
            raise ValueError(f"Máximo {spec.max_length} caracteres.")

    if spec.options and not spec.dynamic_options and value not in {o.value for o in spec.options}:
        raise ValueError("Opción no válida.")
    return value


def _fail(message: str) -> Any:
    raise ValueError(message)


def random_seed() -> int:
    return random.randint(0, SEED_MAX)
