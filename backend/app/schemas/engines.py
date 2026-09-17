from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.engines.base import EngineCapabilities, EngineVariant, LicenseInfo, ParameterSpec


class EngineSummary(BaseModel):
    id: str
    name: str
    description: str
    installed: bool
    missing_packages: list[str]
    implemented: bool = Field(description="La generación está disponible en esta versión")
    implementation_phase: int | None
    license: LicenseInfo
    supports_custom_checkpoints: bool = False
    variants: list[EngineVariant]
    default_variant: str


class PresetInfo(BaseModel):
    id: str
    label: str
    values: dict[str, Any]


class EngineConfig(BaseModel):
    engine: EngineSummary
    variant: EngineVariant
    capabilities: EngineCapabilities
    parameters: list[ParameterSpec]
    presets: list[PresetInfo]


class ValidateRequest(BaseModel):
    variant: str | None = None
    params: dict[str, Any] = {}


class ValidateResponse(BaseModel):
    variant: str
    params: dict[str, Any]
