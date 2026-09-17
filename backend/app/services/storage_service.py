"""Disk usage by category and safe cleanup of regenerable caches, temporary files and orphaned files.

Never removes anything referenced by the database: reference originals/processed audio still used by a reference,
generation outputs, transcripts or profiles. Orphans younger than ORPHAN_MIN_AGE_S are kept, so files being written
by a running job are never touched.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import tempfile
import time
from pathlib import Path

from pydantic import BaseModel
from sqlmodel import Session, select

from app.core.config import Settings
from app.models.entities import Generation, GenerationSegment, ReferenceAudio
from app.workers.asyncio_queue import AsyncioJobQueue

logger = logging.getLogger("voicelab.storage")

ORPHAN_MIN_AGE_S = 600
UPLOAD_MIN_AGE_S = 3600


class StorageCategory(BaseModel):
    id: str
    label: str
    bytes: int
    files: int
    removable: bool
    description: str


class StorageUsage(BaseModel):
    categories: list[StorageCategory]
    orphans_bytes: int
    orphans_files: int
    total_bytes: int


class CleanupRequest(BaseModel):
    orphans: bool = True
    temp: bool = True
    reference_clips: bool = False
    transcript_cache: bool = False


class CleanupReport(BaseModel):
    removed_files: int
    freed_bytes: int
    details: dict[str, int]
    skipped: list[str]


def _size(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    if path.is_file():
        return path.stat().st_size, 1
    total = count = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
                count += 1
            except OSError:
                pass
    return total, count


def _old_enough(path: Path, min_age_s: float) -> bool:
    try:
        return time.time() - path.stat().st_mtime >= min_age_s
    except OSError:
        return False


class StorageService:
    def __init__(self, session: Session, settings: Settings, queue: AsyncioJobQueue) -> None:
        self.session, self.settings, self.queue = session, settings, queue
        self.data = settings.data_dir

    # ------------------------------------------------------------------ orphans
    def _orphans(self) -> list[Path]:
        root = self.data.resolve()
        found: list[Path] = []

        shas = {sha for sha in self.session.exec(select(ReferenceAudio.sha256))}
        for sub in ("original", "processed", "analysis"):
            folder = self.data / "references" / sub
            if folder.is_dir():
                found += [f for f in folder.iterdir() if f.is_file() and f.name.split(".")[0] not in shas]

        gens = list(self.session.exec(select(Generation)))
        ids = {g.id for g in gens}
        used = {(root / p).resolve() for g in gens for p in (g.output_path, g.raw_output_path) if p}
        used |= {(root / s).resolve() for s in self.session.exec(select(GenerationSegment.audio_path)) if s}
        generated = self.data / "generated"
        if generated.is_dir():
            for entry in generated.iterdir():
                if entry.is_dir():
                    if entry.name not in ids:
                        found.append(entry)
                    else:  # superseded final files (e.g. kept open by a player while re-mastering)
                        found += [f for f in entry.glob("final_*.wav") if f.resolve() not in used]
                elif entry.is_file() and entry.resolve() not in used:
                    found.append(entry)
        return [p for p in found if _old_enough(p, ORPHAN_MIN_AGE_S)]

    def _temp_files(self) -> list[Path]:
        found = [f for f in self.data.rglob("*.tmp.wav") if _old_enough(f, ORPHAN_MIN_AGE_S)]
        uploads = self.data / "cache" / "uploads"
        if uploads.is_dir():
            found += [f for f in uploads.iterdir() if _old_enough(f, UPLOAD_MIN_AGE_S)]
        found += [d for d in Path(tempfile.gettempdir()).glob("voicelab_pp_*") if _old_enough(d, ORPHAN_MIN_AGE_S)]
        return found

    # ------------------------------------------------------------------ usage
    def usage(self) -> StorageUsage:
        spec = [
            ("references_original", "Audios originales", self.data / "references" / "original", False,
             "Archivos subidos, sin modificar."),
            ("references_processed", "Audios procesados", self.data / "references" / "processed", False,
             "WAV 24 kHz mono usados por los modelos."),
            ("generated", "Audios generados", self.data / "generated", False,
             "Salida original de los modelos, segmentos y versión final."),
            ("projects", "Exportaciones de proyectos", self.data / "projects", True,
             "Último audio exportado de cada proyecto; se vuelve a crear al exportar."),
            ("reference_clips", "Recortes de referencia", self.data / "cache" / "references", True,
             "Caché: se regenera automáticamente al generar."),
            ("transcript_cache", "Caché de transcripciones", self.data / "cache" / "transcripts", True,
             "Caché de Whisper: las transcripciones guardadas no se pierden."),
            ("uploads", "Subidas temporales", self.data / "cache" / "uploads", True, "Archivos en tránsito."),
            ("database", "Base de datos", self.data / "database", False, "Perfiles, referencias y generaciones."),
            ("voices", "Manifiestos de voces", self.data / "voices", False, "profile.json de cada perfil."),
            ("logs", "Registros", self.data / "logs", False, "Logs del servidor."),
            ("models", "Pesos de modelos", self.settings.hf_home, False,
             "Descargados desde Hugging Face. Se gestionan desde Modelos."),
        ]
        categories = []
        for cid, label, path, removable, description in spec:
            size, count = _size(path)
            categories.append(StorageCategory(id=cid, label=label, bytes=size, files=count, removable=removable,
                                              description=description))
        orphan_bytes = orphan_files = 0
        for path in self._orphans():
            size, count = _size(path)
            orphan_bytes += size
            orphan_files += count
        return StorageUsage(categories=categories, orphans_bytes=orphan_bytes, orphans_files=orphan_files,
                            total_bytes=sum(c.bytes for c in categories))

    # ------------------------------------------------------------------ cleanup
    def cleanup(self, body: CleanupRequest) -> CleanupReport:
        details: dict[str, int] = {}
        skipped: list[str] = []
        removed = freed = 0
        running, waiting = self.queue.counts()
        busy = bool(running or waiting)

        groups: dict[str, list[Path]] = {}
        if body.orphans:
            groups["orphans"] = self._orphans()
        if body.temp:
            groups["temp"] = self._temp_files()
        for flag, sub in (("reference_clips", ("cache", "references")), ("transcript_cache", ("cache", "transcripts"))):
            if not getattr(body, flag):
                continue
            if busy and flag == "reference_clips":
                skipped.append("Los recortes de referencia no se borran mientras hay generaciones en curso.")
                continue
            folder = self.data.joinpath(*sub)
            groups[flag] = [p for p in folder.iterdir()] if folder.is_dir() else []

        root = self.data.resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        for name, paths in groups.items():
            count_before = removed
            for path in paths:
                resolved = path.resolve()
                if root not in resolved.parents and temp_root not in resolved.parents:
                    continue  # never outside the data folder / system temp
                size, count = _size(path)
                try:
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)  # originals are stored read-only
                        path.unlink()
                except OSError:
                    continue  # in use (Windows); try again later
                removed += count
                freed += size
            details[name] = removed - count_before
        logger.info("storage_cleanup", extra={"removed": removed, "freed_bytes": freed, **details})
        return CleanupReport(removed_files=removed, freed_bytes=freed, details=details, skipped=skipped)
