"""Content-addressed storage for reference audio.

data/references/original/<sha256>.<ext>   immutable (read-only) upload
data/references/processed/<sha256>.wav    24 kHz mono float32 working copy
data/references/analysis/<sha256>.json    analysis + provenance
Paths are built only from validated hashes/extensions, never from user filenames.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.audio.validation import ALLOWED_FORMATS
from app.core.errors import AppError, ErrorCode

CHUNK = 1024 * 1024
_SHA = re.compile(r"^[0-9a-f]{64}$")


def _check_sha(sha: str) -> str:
    if not _SHA.match(sha):
        raise ValueError("invalid sha256")
    return sha


class AudioStorage:
    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir.resolve()
        self.original_dir = self.root / "references" / "original"
        self.processed_dir = self.root / "references" / "processed"
        self.analysis_dir = self.root / "references" / "analysis"
        self.tmp_dir = self.root / "cache" / "uploads"
        for d in (self.original_dir, self.processed_dir, self.analysis_dir, self.tmp_dir):
            d.mkdir(parents=True, exist_ok=True)

    def original_path(self, sha: str, ext: str) -> Path:
        if ext not in ALLOWED_FORMATS:
            raise ValueError("invalid extension")
        return self.original_dir / f"{_check_sha(sha)}.{ext}"

    def processed_path(self, sha: str) -> Path:
        return self.processed_dir / f"{_check_sha(sha)}.wav"

    def analysis_path(self, sha: str) -> Path:
        return self.analysis_dir / f"{_check_sha(sha)}.json"

    def temp_path(self, suffix: str = "") -> Path:
        return self.tmp_dir / f"{uuid.uuid4().hex}{suffix}"

    async def save_upload(self, upload: UploadFile, max_bytes: int) -> tuple[Path, str, int]:
        """Stream to a temp file while hashing; abort as soon as the size limit is exceeded."""
        tmp = self.temp_path(".upload")
        digest, size = hashlib.sha256(), 0
        try:
            with tmp.open("wb") as fh:
                while chunk := await upload.read(CHUNK):
                    size += len(chunk)
                    if size > max_bytes:
                        raise AppError(ErrorCode.FILE_TOO_LARGE, status_code=413,
                                       details={"maximo_mb": max_bytes // (1024 * 1024)})
                    digest.update(chunk)
                    fh.write(chunk)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        if size == 0:
            tmp.unlink(missing_ok=True)
            raise AppError(ErrorCode.AUDIO_DECODE_ERROR, status_code=422, message="El archivo está vacío.")
        return tmp, digest.hexdigest(), size

    def commit_original(self, tmp: Path, sha: str, ext: str) -> Path:
        dst = self.original_path(sha, ext)
        if dst.exists():
            tmp.unlink(missing_ok=True)
        else:
            tmp.replace(dst)
            os.chmod(dst, stat.S_IREAD)  # originals are immutable
        return dst

    def find_original(self, sha: str) -> Path | None:
        return next(iter(sorted(self.original_dir.glob(f"{_check_sha(sha)}.*"))), None)

    def delete_all(self, sha: str) -> None:
        for path in (*self.original_dir.glob(f"{_check_sha(sha)}.*"), self.processed_path(sha),
                     self.analysis_path(sha)):
            if path.exists():
                os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
                path.unlink()
