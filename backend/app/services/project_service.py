"""Projects: a script split into ordered segments, generated with shared defaults and exported as one audio.

Each segment inherits the project settings (engine, variant, parameters, voice, markup) and may override voice,
emotion, pause and seed. A hash of the effective settings is stored when a segment is generated, so later edits
mark it as "stale" instead of silently exporting audio that no longer matches the script.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from sqlmodel import Session, select

from app.core.errors import AppError, ErrorCode
from app.generation.assembly import SegmentAudio, assemble
from app.generation.planner import split_long_text
from app.models.entities import Generation, Project, ProjectSegment
from app.models.enums import JobStatus
from app.postprocess.processor import process
from app.schemas.generation import GenerationCreate
from app.schemas.project import (
    MAX_SEGMENTS,
    ProjectCreate,
    ProjectGenerate,
    ProjectRead,
    ProjectSettings,
    ProjectSummary,
    ProjectUpdate,
    SegmentRead,
    SegmentsAdd,
    SegmentStatus,
    SegmentUpdate,
)
from app.services import mastering
from app.services.generation_service import GenerationService
from app.voice_profiles.emotions import EMOTIONS

logger = logging.getLogger("voicelab.projects")

SENTENCE_CHUNK_CHARS = 280  # "sentences" import: short, natural chunks that are quick to regenerate one by one
EXPORT_SUBTYPE = "PCM_24"
_OVERRIDES = ("profile_id", "reference_id", "emotion", "intensity", "pause_after_ms", "seed")


def split_script(text: str, mode: str) -> list[str]:
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n|\r?\n", text) if p.strip()]
    if mode == "paragraphs":
        return paragraphs
    out: list[str] = []
    for paragraph in paragraphs:
        out += [s for s in re.split(r"(?<=[.!?…])\s+", paragraph) if s.strip()]
    return [chunk for s in out for chunk in split_long_text(s, SENTENCE_CHUNK_CHARS)]


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE).strip("_")[:60] or "proyecto"


class ProjectService:
    def __init__(self, session: Session, generations: GenerationService) -> None:
        self.session = session
        self.generations = generations
        self.settings = generations.settings

    # ------------------------------------------------------------------ helpers
    def get(self, project_id: str) -> Project:
        project = self.session.get(Project, project_id)
        if project is None:
            raise AppError(ErrorCode.PROJECT_NOT_FOUND, status_code=404)
        return project

    def _segment(self, project: Project, segment_id: str) -> ProjectSegment:
        seg = self.session.get(ProjectSegment, segment_id)
        if seg is None or seg.project_id != project.id:
            raise AppError(ErrorCode.PROJECT_NOT_FOUND, status_code=404, message="El segmento no existe.")
        return seg

    def segments_of(self, project_id: str) -> list[ProjectSegment]:
        stmt = select(ProjectSegment).where(ProjectSegment.project_id == project_id).order_by(
            ProjectSegment.position)  # type: ignore[arg-type]
        return list(self.session.exec(stmt))

    @staticmethod
    def project_settings(project: Project) -> ProjectSettings:
        return ProjectSettings.model_validate(project.settings or {})

    def _touch(self, project: Project) -> None:
        project.updated_at = datetime.now(UTC)
        self.session.add(project)

    def _renumber(self, segments: list[ProjectSegment]) -> None:
        for i, seg in enumerate(segments):
            if seg.position != i:
                seg.position = i
                self.session.add(seg)

    # ------------------------------------------------------------------ effective settings
    def body_for(self, project: Project, seg: ProjectSegment) -> GenerationCreate:
        cfg = self.project_settings(project)
        if not cfg.engine:
            raise AppError(ErrorCode.VALIDATION_ERROR, status_code=422,
                           message="Elige el motor de voz del proyecto antes de generar.")
        params = dict(cfg.params)
        if seg.seed is not None:
            params["seed"] = seg.seed
        profile_id = seg.profile_id if seg.profile_id is not None else cfg.profile_id
        reference_id = seg.reference_id if seg.reference_id is not None else (
            cfg.reference_id if seg.profile_id is None else None)  # a voice override uses that profile's reference
        return GenerationCreate(engine=cfg.engine, variant=cfg.variant, text=seg.text, params=params,
                                profile_id=profile_id, reference_id=reference_id, emotion=seg.emotion,
                                intensity=seg.intensity if seg.intensity is not None else 50, markup=cfg.markup,
                                normalize=cfg.normalize)

    @staticmethod
    def signature(body: GenerationCreate) -> str:
        # normalize=True (default) is left out so projects generated before it existed keep their hash
        exclude = {"preview", "postprocess"} | ({"normalize"} if body.normalize else set())
        data = body.model_dump(exclude=exclude)
        return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def _status(self, project: Project, seg: ProjectSegment, gen: Generation | None) -> SegmentStatus:
        if gen is None:
            return "empty"
        if gen.status == JobStatus.QUEUED:
            return "queued"
        if not gen.status.is_terminal:
            return "generating"
        if gen.status == JobStatus.FAILED:
            return "failed"
        if gen.status == JobStatus.CANCELLED:
            return "cancelled"
        try:
            current = self.signature(self.body_for(project, seg))
        except AppError:
            current = None
        return "ready" if current == seg.generated_signature else "stale"

    # ------------------------------------------------------------------ projects
    def list(self) -> list[ProjectSummary]:
        out = []
        stmt = select(Project).order_by(Project.updated_at.desc())  # type: ignore[attr-defined]
        for project in self.session.exec(stmt):
            counts = {"ready": 0, "pending": 0, "running": 0, "failed": 0}
            segments = self.segments_of(project.id)
            for seg in segments:
                gen = self.session.get(Generation, seg.generation_id) if seg.generation_id else None
                status = self._status(project, seg, gen)
                key = {"ready": "ready", "queued": "running", "generating": "running", "failed": "failed"}.get(
                    status, "pending")
                counts[key] += 1
            out.append(ProjectSummary(id=project.id, name=project.name, description=project.description,
                                      segments=len(segments), updated_at=project.updated_at, **counts))
        return out

    def create(self, body: ProjectCreate) -> Project:
        project = Project(name=body.name.strip(), description=body.description,
                          settings=body.settings.model_dump())
        self.session.add(project)
        self.session.commit()
        self.session.refresh(project)
        return project

    def update(self, project_id: str, body: ProjectUpdate) -> Project:
        project = self.get(project_id)
        if body.name is not None:
            project.name = body.name.strip()
        if "description" in body.model_fields_set:
            project.description = body.description
        if body.settings is not None:
            project.settings = body.settings.model_dump()
        self._touch(project)
        self.session.commit()
        self.session.refresh(project)
        return project

    async def delete(self, project_id: str) -> None:
        project = self.get(project_id)
        for seg in self.segments_of(project.id):
            await self._drop_generation(seg)
            self.session.delete(seg)
        folder = self.settings.data_dir / "projects" / project.id
        shutil.rmtree(folder, ignore_errors=True)
        self.session.delete(project)
        self.session.commit()

    async def _drop_generation(self, seg: ProjectSegment) -> None:
        if seg.generation_id and self.session.get(Generation, seg.generation_id) is not None:
            await self.generations.delete(seg.generation_id)
        seg.generation_id = None
        seg.generated_signature = None

    # ------------------------------------------------------------------ segments
    def add_segments(self, project_id: str, body: SegmentsAdd) -> Project:
        project = self.get(project_id)
        segments = self.segments_of(project.id)
        if len(segments) + len(body.segments) > MAX_SEGMENTS:
            raise AppError(ErrorCode.VALIDATION_ERROR, status_code=422,
                           message=f"Un proyecto admite como máximo {MAX_SEGMENTS} segmentos.")
        at = len(segments) if body.position is None else min(body.position, len(segments))
        new = [ProjectSegment(project_id=project.id, text=s.text) for s in body.segments]
        for seg in new:
            self.session.add(seg)
        self._renumber(segments[:at] + new + segments[at:])
        self._touch(project)
        self.session.commit()
        return project

    def import_script(self, project_id: str, text: str, mode: str) -> Project:
        from app.schemas.project import SegmentInput

        parts = split_script(text, mode)
        if not parts:
            raise AppError(ErrorCode.VALIDATION_ERROR, status_code=422, message="El guion no contiene texto.")
        return self.add_segments(project_id, SegmentsAdd(segments=[SegmentInput(text=p) for p in parts]))

    def update_segment(self, project_id: str, segment_id: str, body: SegmentUpdate) -> Project:
        project = self.get(project_id)
        seg = self._segment(project, segment_id)
        if body.emotion is not None and body.emotion not in EMOTIONS:
            raise AppError(ErrorCode.VALIDATION_ERROR, status_code=422, message="Emoción no válida.")
        if body.text is not None:
            seg.text = body.text.strip()
        for field in _OVERRIDES:
            if field in body.model_fields_set:
                setattr(seg, field, getattr(body, field))
        seg.updated_at = datetime.now(UTC)
        self.session.add(seg)
        self._touch(project)
        self.session.commit()
        return project

    async def delete_segment(self, project_id: str, segment_id: str) -> Project:
        project = self.get(project_id)
        seg = self._segment(project, segment_id)
        await self._drop_generation(seg)
        self.session.delete(seg)
        self.session.flush()
        self._renumber(self.segments_of(project.id))
        self._touch(project)
        self.session.commit()
        return project

    def reorder(self, project_id: str, ids: list[str]) -> Project:
        project = self.get(project_id)
        segments = {s.id: s for s in self.segments_of(project.id)}
        if sorted(ids) != sorted(segments):
            raise AppError(ErrorCode.VALIDATION_ERROR, status_code=422,
                           message="El nuevo orden debe incluir todos los segmentos exactamente una vez.")
        self._renumber([segments[i] for i in ids])
        self._touch(project)
        self.session.commit()
        return project

    # ------------------------------------------------------------------ generation
    async def generate(self, project_id: str, body: ProjectGenerate) -> Project:
        project = self.get(project_id)
        segments = self.segments_of(project.id)
        if body.segment_ids is not None:
            wanted = set(body.segment_ids)
            segments = [s for s in segments if s.id in wanted]
        queue: list[tuple[ProjectSegment, GenerationCreate]] = []
        for seg in segments:
            gen = self.session.get(Generation, seg.generation_id) if seg.generation_id else None
            status = self._status(project, seg, gen)
            if status in ("queued", "generating") or (body.only_pending and status == "ready"):
                continue
            request = self.body_for(project, seg)
            try:
                self.generations.build(request)  # validate everything before touching any segment
            except AppError as err:
                extra = {"segmento": seg.position + 1}
                err.details = {**err.details, **extra} if isinstance(err.details, dict) else {
                    **extra, **({"errores": err.details} if err.details is not None else {})}
                raise
            queue.append((seg, request))
        for seg, request in queue:
            await self._drop_generation(seg)
            gen = await self.generations.create(request, kind="project", project_id=project.id,
                                                label=f"Segmento {seg.position + 1}")
            seg.generation_id = gen.id
            seg.generated_signature = self.signature(request)
            self.session.add(seg)
        self._touch(project)
        self.session.commit()
        return project

    # ------------------------------------------------------------------ export
    def _ready_parts(self, project: Project, allow_partial: bool) -> list[tuple[ProjectSegment, Path]]:
        parts, missing = [], []
        for seg in self.segments_of(project.id):
            gen = self.session.get(Generation, seg.generation_id) if seg.generation_id else None
            status = self._status(project, seg, gen)
            if status == "ready" and gen is not None:
                parts.append((seg, self.generations.audio_path(gen)))
            elif status == "stale" and gen is not None and allow_partial:
                parts.append((seg, self.generations.audio_path(gen)))
            else:
                missing.append(seg.position + 1)
        if missing and not allow_partial:
            raise AppError(ErrorCode.PROJECT_NOT_READY, status_code=409, details={"segmentos": missing})
        if not parts:
            raise AppError(ErrorCode.PROJECT_NOT_READY, status_code=409, details={"segmentos": missing},
                           message="No hay ningún segmento generado para exportar.")
        return parts

    def render(self, project_id: str, allow_partial: bool = False) -> tuple[Path, list[tuple[ProjectSegment, Path]]]:
        project = self.get(project_id)
        cfg = self.project_settings(project)
        parts = self._ready_parts(project, allow_partial)
        audio_parts = []
        for i, (seg, path) in enumerate(parts):
            data, sr = sf.read(path, dtype="float32", always_2d=False)
            last = i == len(parts) - 1
            pause = 0 if last else (seg.pause_after_ms if seg.pause_after_ms is not None else cfg.default_pause_ms)
            audio_parts.append(SegmentAudio(np.asarray(data).reshape(-1), sr, 0, pause))
        audio, sr = assemble(audio_parts)
        if cfg.export_postprocess and cfg.export_postprocess.dsp_active:
            audio, _report = process(audio, sr, cfg.export_postprocess, mastering.ffmpeg_or_none(self.settings))
        if audio.size and float(np.max(np.abs(audio))) > 1.0:
            audio = np.clip(audio, -1.0, 1.0)
        folder = self.settings.data_dir / "projects" / project.id
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / "export.wav"
        tmp = folder / "export.tmp.wav"
        sf.write(tmp, audio, sr, subtype=EXPORT_SUBTYPE)
        tmp.replace(target)
        project.exported_at = datetime.now(UTC)
        self.session.add(project)
        self.session.commit()
        return target, parts

    def export_zip(self, project_id: str, allow_partial: bool = False) -> bytes:
        wav, parts = self.render(project_id, allow_partial)
        project = self.get(project_id)
        cfg = self.project_settings(project)
        buffer = io.BytesIO()
        manifest = {"format": "voicelab-project", "version": 1, "name": project.name,
                    "description": project.description, "settings": cfg.model_dump(), "segments": []}
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(wav, f"{_safe_name(project.name)}.wav")
            for seg, path in parts:
                gen = self.session.get(Generation, seg.generation_id)
                name = f"segmentos/{seg.position + 1:03d}.wav"
                archive.write(path, name)
                manifest["segments"].append({
                    "position": seg.position + 1, "text": seg.text, "file": name,
                    "engine": gen.engine if gen else None, "variant": gen.variant if gen else None,
                    "seed": gen.seed if gen else None, "params": gen.params if gen else None,
                    "reference": (gen.reference_snapshot or {}).get("name") if gen else None,
                    "emotion": seg.emotion, "pause_after_ms": seg.pause_after_ms,
                    "duration_s": gen.duration_s if gen else None,
                })
            archive.writestr("project.json", json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
        return buffer.getvalue()

    # ------------------------------------------------------------------ read models
    async def to_read(self, project: Project) -> ProjectRead:
        cfg = self.project_settings(project)
        segments = []
        total = 0.0
        for seg in self.segments_of(project.id):
            gen = self.session.get(Generation, seg.generation_id) if seg.generation_id else None
            status = self._status(project, seg, gen)
            if gen is not None and gen.duration_s:
                total += gen.duration_s
            segments.append(SegmentRead(
                id=seg.id, position=seg.position, text=seg.text, profile_id=seg.profile_id,
                reference_id=seg.reference_id, emotion=seg.emotion, intensity=seg.intensity,
                pause_after_ms=seg.pause_after_ms, seed=seg.seed,
                effective_pause_ms=seg.pause_after_ms if seg.pause_after_ms is not None else cfg.default_pause_ms,
                status=status, generation=await self.generations.to_read(gen) if gen else None))
        return ProjectRead(id=project.id, name=project.name, description=project.description, settings=cfg,
                           segments=segments, total_duration_s=round(total, 3), exported_at=project.exported_at,
                           created_at=project.created_at, updated_at=project.updated_at)

    def export_filename(self, project_id: str, extension: str) -> str:
        return f"{_safe_name(self.get(project_id).name)}.{extension}"
