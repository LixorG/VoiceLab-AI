from __future__ import annotations

from app.engines.base import EngineVariant, LicenseInfo
from app.engines.flow_matching import FlowMatchingBackend


class E2TTSBackend(FlowMatchingBackend):
    id = "e2tts"
    display_name = "E2-TTS"
    description = ("Reproducción comunitaria del paper E2 TTS (Microsoft) incluida en F5-TTS. "
                   "Mismos parámetros que F5-TTS con otra arquitectura.")
    license = LicenseInfo(
        code="MIT", weights="CC-BY-NC-4.0", commercial_use="no_permitido",
        notes="Microsoft no publicó código ni pesos oficiales; se usa E2TTS_Base del repositorio F5-TTS.",
        url="https://arxiv.org/abs/2406.18009",
    )
    checkpoints = {"E2TTS_Base": ("SWivid/E2-TTS", "E2TTS_Base/model_1200000.safetensors")}

    def builtin_variants(self) -> list[EngineVariant]:
        return [
            EngineVariant(id="E2TTS_Base", label="E2-TTS Base",
                          description="Transformer plano con conexiones tipo U-Net, entrenado en Emilia.",
                          repo_id="SWivid/E2-TTS", vram_estimate_mb=3000, download_size_mb=1330),
        ]


ENGINE_CLASS = E2TTSBackend
