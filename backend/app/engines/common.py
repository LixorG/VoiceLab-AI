"""Helpers shared by engine adapters (reference clips, Hugging Face cache)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.core.errors import AppError, ErrorCode
from app.engines.base import ReferenceInput


def reference_clip(reference: ReferenceInput) -> tuple[Path, str]:
    """Cut the selected range of the processed 24 kHz WAV into a cached clip. Returns (path, cache key)."""
    import soundfile as sf

    from app.core.config import get_settings

    key = hashlib.sha256(f"{reference.sha256}|{reference.start_s}|{reference.end_s}".encode()).hexdigest()
    cache_dir = get_settings().data_dir / "cache" / "references" / "clips"
    cache_dir.mkdir(parents=True, exist_ok=True)
    clip = cache_dir / f"{key}.wav"
    if not clip.exists():
        if not Path(reference.audio_path).exists():
            raise AppError(ErrorCode.NOT_FOUND, status_code=404,
                           message="El audio procesado de la referencia no existe.")
        info = sf.info(reference.audio_path)
        start = int((reference.start_s or 0) * info.samplerate)
        stop = int(reference.end_s * info.samplerate) if reference.end_s is not None else None
        audio, sr = sf.read(reference.audio_path, start=start, stop=stop, dtype="float32", always_2d=False)
        tmp = clip.with_suffix(".tmp.wav")
        sf.write(tmp, audio, sr, subtype="PCM_16")
        tmp.replace(clip)
    return clip, key


def prepare_hf_cache() -> None:
    """Work around a huggingface_hub race on Windows without symlink rights.

    `are_symlinks_supported` marks a directory as supported *before* testing it, so parallel downloads can try
    to create symlinks and fail with WinError 1314. Running the check once up front avoids that.
    """
    from huggingface_hub import constants, file_download

    file_download.are_symlinks_supported(constants.HF_HUB_CACHE)


def hf_files_cached(repo_id: str, filenames: tuple[str, ...]) -> bool:
    from huggingface_hub import try_to_load_from_cache

    return all(isinstance(try_to_load_from_cache(repo_id, f), str) for f in filenames)
