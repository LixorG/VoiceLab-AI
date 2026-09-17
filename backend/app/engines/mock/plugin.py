"""Simulated engine for development and tests (ENABLE_MOCK_ENGINE=true). Produces a tone, not speech."""

from __future__ import annotations

import numpy as np

from app.engines.base import (
    CancelToken,
    ControlCapability,
    ControlSource,
    EngineCapabilities,
    EngineRequest,
    EngineResult,
    EngineVariant,
    GenericControl,
    LicenseInfo,
    ParameterSpec,
    ParameterTooltip,
    PreparedReference,
    ProgressCallback,
    ReferenceInput,
    TTSBackend,
    random_seed,
)

SR = 24_000


class MockBackend(TTSBackend):
    id = "mock"
    display_name = "Motor simulado"
    description = "Motor de pruebas: genera un tono, no voz. Solo para desarrollo."
    license = LicenseInfo(code="MIT", weights="—", commercial_use="permitido", url="https://example.invalid")
    implementation_phase = None

    def __init__(self) -> None:
        self._loaded: str | None = None

    def variants(self) -> list[EngineVariant]:
        return [EngineVariant(id="tone", label="Tono", description="Onda senoidal", vram_estimate_mb=0,
                              download_size_mb=0)]

    def capabilities(self, variant: str | None = None) -> EngineCapabilities:
        v = self.variant(variant)
        na = ControlCapability(source=ControlSource.UNAVAILABLE, reason="El motor simulado no lo admite.")
        return EngineCapabilities(
            variant=v.id, mode="clone", sample_rate=SR, languages=["es", "en"],
            requires_reference_audio=False, requires_reference_text=False,
            reference_strategies=["best_reference"], deterministic_seed=True,
            controls={
                GenericControl.SPEED: ControlCapability(source=ControlSource.NATIVE, parameter="speed"),
                GenericControl.SEED: ControlCapability(source=ControlSource.NATIVE, parameter="seed"),
                GenericControl.PITCH: ControlCapability(source=ControlSource.NATIVE, parameter="tone_hz"),
                **{c: na for c in (GenericControl.EMOTION, GenericControl.NATURALNESS, GenericControl.VOICE_COLOR,
                                   GenericControl.INSTRUCTION, GenericControl.TARGET_DURATION,
                                   GenericControl.LANGUAGE)},
            },
        )

    def get_parameters_schema(self, variant: str | None = None) -> list[ParameterSpec]:
        tip = ParameterTooltip(what="Parámetro de prueba.", effect="—", cost="—", typical="—")
        return [
            ParameterSpec(id="speed", type="slider", value_type="float", label="Velocidad", default=1.0,
                          min=0.5, max=2.0, step=0.1, group="basic", control=GenericControl.SPEED, tooltip=tip),
            ParameterSpec(id="tone_hz", type="slider", value_type="int", label="Frecuencia", default=220, min=80,
                          max=880, step=10, unit="Hz", group="basic", control=GenericControl.PITCH, tooltip=tip,
                          presets={"fast": 110, "balanced": 220, "high_fidelity": 440}),
            ParameterSpec(id="seed", type="seed", value_type="int", label="Semilla", control=GenericControl.SEED,
                          tooltip=tip),
        ]

    def load(self, variant: str, device: str, dtype: str) -> None:
        self._loaded = variant

    def unload(self) -> None:
        self._loaded = None

    @property
    def loaded_variant(self) -> str | None:
        return self._loaded

    def prepare_reference(self, reference: ReferenceInput, variant: str) -> PreparedReference:
        return PreparedReference(engine=self.id, variant=variant, cache_key=reference.sha256)

    def generate(self, request: EngineRequest, progress: ProgressCallback, cancel: CancelToken) -> EngineResult:
        params = self.validate_parameters(request.params, request.variant)
        seed = params["seed"] if params["seed"] is not None else random_seed()
        rng = np.random.default_rng(seed)
        duration = max(0.3, len(request.text) * 0.06 / params["speed"])
        t = np.arange(int(duration * SR)) / SR
        audio = (0.2 * np.sin(2 * np.pi * params["tone_hz"] * t) + rng.normal(0, 0.002, t.size)).astype(np.float32)
        progress(1.0, None)
        return EngineResult(audio=audio, sample_rate=SR, seed=seed, effective_params={**params, "seed": seed})


ENGINE_CLASS = MockBackend
