from __future__ import annotations

from app.engines.base import EngineVariant, LicenseInfo
from app.engines.flow_matching import FlowMatchingBackend


class F5TTSBackend(FlowMatchingBackend):
    id = "f5tts"
    display_name = "F5-TTS"
    description = "Clonación de voz por flow matching (Diffusion Transformer). Necesita audio y texto de referencia."
    license = LicenseInfo(
        code="MIT", weights="CC-BY-NC-4.0", commercial_use="no_permitido",
        notes="Pesos no comerciales por el dataset de entrenamiento (Emilia).",
        url="https://github.com/SWivid/F5-TTS",
    )

    checkpoints = {
        "F5TTS_v1_Base": ("SWivid/F5-TTS", "F5TTS_v1_Base/model_1250000.safetensors"),
        "F5TTS_Base": ("SWivid/F5-TTS", "F5TTS_Base/model_1200000.safetensors"),
    }

    def builtin_variants(self) -> list[EngineVariant]:
        return [
            EngineVariant(id="F5TTS_v1_Base", label="F5-TTS v1 Base",
                          description="Checkpoint oficial recomendado (v1).",
                          repo_id="SWivid/F5-TTS", vram_estimate_mb=3000, download_size_mb=1340),
            EngineVariant(id="F5TTS_Base", label="F5-TTS Base (original)",
                          description="Primer checkpoint publicado; se conserva por compatibilidad.",
                          repo_id="SWivid/F5-TTS", vram_estimate_mb=3000, download_size_mb=1340),
        ]


ENGINE_CLASS = F5TTSBackend
