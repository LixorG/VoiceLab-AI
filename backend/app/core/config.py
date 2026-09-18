"""Application settings loaded from environment / .env."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# voice-lab/ (repo root): backend/app/core/config.py -> parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_SUBDIRS = (
    "voices",
    "references/original",
    "references/processed",
    "references/analysis",
    "generated",
    "cache",
    "database",
    "logs",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", PROJECT_ROOT / "backend" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["development", "production", "test"] = "development"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    data_dir: Path = Path("./data")
    model_dir: Path = Path("./models")
    hf_home: Path = Path("./models/huggingface")
    database_url: str | None = None

    device: Literal["auto", "cuda", "rocm", "mps", "cpu"] = "auto"
    default_model: str = "f5tts"
    enable_mock_engine: bool = False  # simulated engine for development/tests
    attn_implementation: Literal["auto", "sdpa", "flash_attention_2", "eager"] = "auto"
    # Qwen3-TTS: replay its per-frame code predictor from CUDA graphs. Measured 2.4-2.8x end to end on the
    # RTX 3080 Laptop with the same WER and speaker similarity; output is equal within bf16 precision, not
    # bit-identical. False = the direct loop (bit-identical, ~1.2x).
    qwen_cuda_graphs: bool = True

    max_upload_mb: int = 500
    max_audio_seconds: int = 600
    # Optional explicit binaries; otherwise resolved from PATH.
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None

    # Free memory automatically after this many idle minutes (0 = never). Never while a job is running or queued.
    model_idle_unload_minutes: int = 15
    asr_idle_unload_minutes: int = 10

    asr_model: str = "large-v3-turbo"
    asr_compute_type: str = "auto"

    # Built UI served by the backend when present (start scripts, Docker). Empty = never serve it.
    frontend_dist: Path | None = PROJECT_ROOT / "frontend" / "dist"

    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    @field_validator("frontend_dist", mode="before")
    @classmethod
    def _optional_dist(cls, value: object) -> object:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        path = Path(str(value))
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    @field_validator("data_dir", "model_dir", "hf_home", mode="after")
    @classmethod
    def _resolve_relative_to_root(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return self.database_url
        db_path = self.data_dir / "database" / "voicelab.db"
        return f"sqlite:///{db_path.as_posix()}"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        for sub in DATA_SUBDIRS:
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def apply_process_environment(self) -> None:
        """Point Hugging Face downloads at MODEL_DIR unless the user already set HF_HOME."""
        os.environ.setdefault("HF_HOME", str(self.hf_home))


@lru_cache
def get_settings() -> Settings:
    return Settings()
