"""Static license information shown in Configuración → Información → Licencias.

Verified in FASE 0 (docs/model-analysis.md). Update when adding engines or dependencies.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class LicenseEntry(BaseModel):
    component: str
    category: Literal["motor", "modelo", "asr", "backend", "frontend", "herramienta"]
    code_license: str | None = None
    weights_license: str | None = None
    commercial_use: Literal["permitido", "no_permitido", "revisar"]
    notes: str | None = None
    url: str


LICENSES: list[LicenseEntry] = [
    LicenseEntry(component="F5-TTS", category="motor", code_license="MIT",
                 weights_license="CC-BY-NC-4.0", commercial_use="no_permitido",
                 notes="Los pesos preentrenados son no comerciales por el dataset Emilia.",
                 url="https://github.com/SWivid/F5-TTS"),
    LicenseEntry(component="E2-TTS (reproducción en F5-TTS)", category="motor", code_license="MIT",
                 weights_license="CC-BY-NC-4.0", commercial_use="no_permitido",
                 notes="Reproducción comunitaria del paper de Microsoft; no hay implementación oficial.",
                 url="https://arxiv.org/abs/2406.18009"),
    LicenseEntry(component="Qwen3-TTS", category="motor", code_license="Apache-2.0",
                 weights_license="Apache-2.0", commercial_use="permitido",
                 url="https://github.com/QwenLM/Qwen3-TTS"),
    LicenseEntry(component="faster-whisper", category="asr", code_license="MIT",
                 weights_license="MIT (Whisper)", commercial_use="permitido",
                 url="https://github.com/SYSTRAN/faster-whisper"),
    LicenseEntry(component="PyTorch", category="backend", code_license="BSD-3-Clause",
                 commercial_use="permitido", url="https://github.com/pytorch/pytorch"),
    LicenseEntry(component="FFmpeg", category="herramienta", code_license="LGPL-2.1+ / GPL según compilación",
                 commercial_use="revisar",
                 notes="La compilación 'full' de Windows incluye componentes GPL.",
                 url="https://ffmpeg.org/legal.html"),
    LicenseEntry(component="FastAPI", category="backend", code_license="MIT",
                 commercial_use="permitido", url="https://github.com/fastapi/fastapi"),
    LicenseEntry(component="SQLModel", category="backend", code_license="MIT",
                 commercial_use="permitido", url="https://github.com/fastapi/sqlmodel"),
    LicenseEntry(component="Pydantic", category="backend", code_license="MIT",
                 commercial_use="permitido", url="https://github.com/pydantic/pydantic"),
    LicenseEntry(component="React", category="frontend", code_license="MIT",
                 commercial_use="permitido", url="https://github.com/facebook/react"),
    LicenseEntry(component="WaveSurfer.js", category="frontend", code_license="BSD-3-Clause",
                 commercial_use="permitido", url="https://github.com/katspaugh/wavesurfer.js"),
    LicenseEntry(component="Tailwind CSS", category="frontend", code_license="MIT",
                 commercial_use="permitido", url="https://github.com/tailwindlabs/tailwindcss"),
    LicenseEntry(component="shadcn/ui", category="frontend", code_license="MIT",
                 commercial_use="permitido", url="https://github.com/shadcn-ui/ui"),
    LicenseEntry(component="Zustand", category="frontend", code_license="MIT",
                 commercial_use="permitido", url="https://github.com/pmndrs/zustand"),
]
