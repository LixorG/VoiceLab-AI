"""Post-processing configuration and report models.

Every processor is disabled by default: nothing touches the model output unless the user enables it explicitly.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DEFAULT_CROSSFADE_MS = 40


class Denoise(BaseModel):
    enabled: bool = False
    strength: Literal["light", "medium", "strong"] = "light"


class TrimSilence(BaseModel):
    enabled: bool = False
    threshold_db: float = Field(default=-45.0, ge=-70.0, le=-20.0, description="Nivel por debajo del cual es silencio")
    padding_ms: int = Field(default=100, ge=0, le=1000, description="Silencio que se conserva en cada extremo")


class TimeStretch(BaseModel):
    enabled: bool = False
    rate: float = Field(default=1.0, ge=0.5, le=2.0, description="Factor de velocidad (1 = sin cambio)")


class PitchShift(BaseModel):
    enabled: bool = False
    semitones: float = Field(default=0.0, ge=-12.0, le=12.0)
    preserve_formants: bool = True


class Loudness(BaseModel):
    enabled: bool = False
    target_lufs: float = Field(default=-16.0, ge=-35.0, le=-10.0)


class Peak(BaseModel):
    enabled: bool = False
    target_dbfs: float = Field(default=-1.0, ge=-12.0, le=0.0)


class Fades(BaseModel):
    enabled: bool = False
    fade_in_ms: int = Field(default=10, ge=0, le=5000)
    fade_out_ms: int = Field(default=30, ge=0, le=5000)


class PostProcessConfig(BaseModel):
    """Chain order is fixed: crossfade (assembly) → denoise → trim → time stretch/pitch → loudness → peak → fades."""

    crossfade_ms: int | None = Field(default=None, ge=0, le=300,
                                     description="Crossfade entre segmentos (solo generaciones segmentadas). "
                                                 "Vacío = 40 ms por defecto.")
    denoise: Denoise = Denoise()
    trim_silence: TrimSilence = TrimSilence()
    time_stretch: TimeStretch = TimeStretch()
    pitch_shift: PitchShift = PitchShift()
    loudness: Loudness = Loudness()
    peak: Peak = Peak()
    fades: Fades = Fades()

    @property
    def dsp_active(self) -> bool:
        return any(p.enabled for p in (self.denoise, self.trim_silence, self.time_stretch, self.pitch_shift,
                                       self.loudness, self.peak, self.fades))

    @property
    def is_active(self) -> bool:
        return self.dsp_active or self.crossfade_ms is not None


class LevelStats(BaseModel):
    duration_s: float
    peak_dbfs: float | None
    loudness_lufs: float | None = Field(description="Sonoridad integrada estimada (EBU R128); vacía si es muy corto")


class ProcessingStep(BaseModel):
    id: str
    label: str
    detail: str


class PostProcessReport(BaseModel):
    steps: list[ProcessingStep]
    warnings: list[str]
    before: LevelStats
    after: LevelStats


class PostProcessCapabilities(BaseModel):
    processors: dict[str, bool]
    reasons: dict[str, str]
