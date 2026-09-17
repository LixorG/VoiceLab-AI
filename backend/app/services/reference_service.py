from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import UploadFile
from sqlmodel import Session, select
from starlette.concurrency import run_in_threadpool

from app.audio.ffmpeg import probe, resolve_binary
from app.audio.pipeline import run_pipeline
from app.audio.segments import snap_segment
from app.audio.storage import AudioStorage
from app.audio.validation import extension_of, sanitize_filename, validate_probe
from app.core.config import Settings
from app.core.errors import AppError, ErrorCode, classify_exception
from app.models.entities import ReferenceAudio, Transcript, VoiceProfile
from app.models.enums import ReferenceStatus
from app.schemas.references import ReferenceRead, ReferenceUpdate, ReferenceUrls
from app.services.transcription_service import to_read as transcript_to_read
from app.services.transcription_service import words_in_range

logger = logging.getLogger("voicelab.references")


class ReferenceService:
    def __init__(self, session: Session, settings: Settings) -> None:
        self.session = session
        self.settings = settings
        self.storage = AudioStorage(settings.data_dir)

    # ---------- queries ----------
    def get(self, reference_id: str) -> ReferenceAudio:
        ref = self.session.get(ReferenceAudio, reference_id)
        if ref is None:
            raise AppError(ErrorCode.NOT_FOUND, status_code=404, message="La referencia no existe.")
        return ref

    def list(self, profile_id: str | None = None) -> list[ReferenceAudio]:
        stmt = select(ReferenceAudio).order_by(ReferenceAudio.created_at)
        if profile_id is not None:
            stmt = stmt.where(ReferenceAudio.profile_id == profile_id)
        return list(self.session.exec(stmt))

    # ---------- commands ----------
    async def upload(self, upload: UploadFile, profile_id: str | None) -> tuple[ReferenceAudio, bool]:
        name = sanitize_filename(upload.filename or "audio")
        ext = extension_of(name)
        if profile_id is not None and self.session.get(VoiceProfile, profile_id) is None:
            raise AppError(ErrorCode.NOT_FOUND, status_code=404, message="El perfil de voz no existe.")
        ffprobe = resolve_binary("ffprobe", self.settings.ffprobe_path)
        ffmpeg = resolve_binary("ffmpeg", self.settings.ffmpeg_path)

        tmp, sha, size = await self.storage.save_upload(upload, self.settings.max_upload_bytes)
        return await self.ingest_temp(tmp, sha, size, name, ext, profile_id, ffprobe, ffmpeg)

    async def ingest_temp(self, tmp, sha: str, size: int, name: str, ext: str, profile_id: str | None,  # noqa: ANN001
                          ffprobe: str | None = None, ffmpeg: str | None = None) -> tuple[ReferenceAudio, bool]:
        """Validate a file already saved to the upload temp dir and run the pipeline (upload and profile import)."""
        ffprobe = ffprobe or resolve_binary("ffprobe", self.settings.ffprobe_path)
        ffmpeg = ffmpeg or resolve_binary("ffmpeg", self.settings.ffmpeg_path)
        try:
            existing = self.session.exec(
                select(ReferenceAudio).where(ReferenceAudio.sha256 == sha,
                                             ReferenceAudio.profile_id == profile_id)
            ).first()
            if existing is not None:
                tmp.unlink(missing_ok=True)
                return existing, True

            info = await run_in_threadpool(probe, tmp, ffprobe)
            validate_probe(ext, info, self.settings.max_audio_seconds)
            original = self.storage.commit_original(tmp, sha, ext)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

        ref = ReferenceAudio(
            profile_id=profile_id, sha256=sha, original_name=name, format=ext, size_bytes=size,
            sample_rate=info.sample_rate, channels=info.channels, duration_s=round(info.duration_s, 3),
            status=ReferenceStatus.PROCESSING,
        )
        self.session.add(ref)
        self.session.commit()
        await self._process(ref, original, info, ffmpeg, force=False)
        self.sync_profile_manifests(profile_id)
        return ref, False

    async def reanalyze(self, reference_id: str) -> ReferenceAudio:
        ref = self.get(reference_id)
        original = self.storage.find_original(ref.sha256)
        if original is None:
            raise AppError(ErrorCode.NOT_FOUND, status_code=404,
                           message="No se encuentra el archivo original.")
        ffprobe = resolve_binary("ffprobe", self.settings.ffprobe_path)
        ffmpeg = resolve_binary("ffmpeg", self.settings.ffmpeg_path)
        info = await run_in_threadpool(probe, original, ffprobe)
        ref.status = ReferenceStatus.PROCESSING
        self.session.add(ref)
        self.session.commit()
        await self._process(ref, original, info, ffmpeg, force=True)
        return ref

    async def _process(self, ref, original, info, ffmpeg: str, force: bool) -> None:  # noqa: ANN001
        try:
            result = await run_in_threadpool(run_pipeline, self.storage, ref.sha256, original, info, ffmpeg,
                                             force)
        except Exception as exc:  # keep the record, mark it failed with a catalog code
            code = exc.code if isinstance(exc, AppError) else classify_exception(exc)
            logger.exception("reference_pipeline_failed", extra={"reference_id": ref.id, "code": code})
            ref.status = ReferenceStatus.FAILED
            ref.analysis = {"error_code": code.value}
        else:
            summary = result.analysis.model_dump(exclude={"speech_regions"})
            summary["normalization_gain_db"] = result.gain_db
            summary["source_codec"] = info.codec
            summary["source_bit_rate"] = info.bit_rate
            ref.analysis = summary
            ref.quality_score = result.analysis.quality_score
            ref.status = ReferenceStatus.ANALYZED
        ref.updated_at = datetime.now(UTC)
        self.session.add(ref)
        self.session.commit()
        self.session.refresh(ref)

    def update(self, reference_id: str, data: ReferenceUpdate) -> ReferenceAudio:
        ref = self.get(reference_id)
        fields = data.model_dump(exclude_unset=True)
        previous_profile = ref.profile_id
        snap = fields.pop("snap_to_words", False)
        if "profile_id" in fields and fields["profile_id"] != ref.profile_id:
            self._move_to_profile(ref, fields["profile_id"])
        fields.pop("profile_id", None)
        if snap and data.segment_start_s is not None and data.segment_end_s is not None:
            full = self.session.exec(select(Transcript).where(
                Transcript.reference_id == ref.id, Transcript.segment_start_s == None)).first()  # noqa: E711
            if full is not None and full.word_timestamps:
                duration = (ref.analysis or {}).get("duration_s") or ref.duration_s or data.segment_end_s
                fields["segment_start_s"], fields["segment_end_s"] = snap_segment(
                    data.segment_start_s, data.segment_end_s, full.word_timestamps, duration)
        if "segment_start_s" in fields:
            end = data.segment_end_s
            if end is not None and ref.duration_s is not None and end > ref.duration_s + 0.05:
                raise AppError(ErrorCode.VALIDATION_ERROR, status_code=422,
                               message="El segmento supera la duración del audio.",
                               details={"duracion_s": ref.duration_s})
        for key, value in fields.items():
            setattr(ref, key, value)
        ref.updated_at = datetime.now(UTC)
        self.session.add(ref)
        self.session.commit()
        self.session.refresh(ref)
        self.sync_profile_manifests(previous_profile, ref.profile_id)
        return ref

    def _move_to_profile(self, ref: ReferenceAudio, profile_id: str | None) -> None:
        if profile_id is not None and self.session.get(VoiceProfile, profile_id) is None:
            raise AppError(ErrorCode.PROFILE_NOT_FOUND, status_code=404)
        duplicate = self.session.exec(select(ReferenceAudio).where(
            ReferenceAudio.sha256 == ref.sha256, ReferenceAudio.profile_id == profile_id,
            ReferenceAudio.id != ref.id)).first()
        if duplicate is not None:
            raise AppError(ErrorCode.VALIDATION_ERROR, status_code=409,
                           message="Ese perfil ya contiene el mismo archivo de audio.")
        self._clear_primary(ref)
        ref.profile_id = profile_id

    def _clear_primary(self, ref: ReferenceAudio) -> None:
        if ref.profile_id is None:
            return
        profile = self.session.get(VoiceProfile, ref.profile_id)
        if profile is not None and profile.primary_reference_id == ref.id:
            profile.primary_reference_id = None
            self.session.add(profile)

    def delete(self, reference_id: str) -> None:
        ref = self.get(reference_id)
        sha, profile_id = ref.sha256, ref.profile_id
        self._clear_primary(ref)
        for tr in self.session.exec(select(Transcript).where(Transcript.reference_id == ref.id)):
            self.session.delete(tr)
        self.session.flush()  # no ORM relationship: delete children before the parent explicitly
        self.session.delete(ref)
        self.session.commit()
        still_used = self.session.exec(select(ReferenceAudio).where(ReferenceAudio.sha256 == sha)).first()
        if still_used is None:
            self.storage.delete_all(sha)
        self.sync_profile_manifests(profile_id)

    def sync_profile_manifests(self, *profile_ids: str | None) -> None:
        from app.engines.registry import get_engine_registry
        from app.services.profile_service import ProfileService

        ids = {pid for pid in profile_ids if pid}
        if not ids:
            return
        profiles = ProfileService(self.session, self.settings, get_engine_registry())
        for pid in ids:
            profile = self.session.get(VoiceProfile, pid)
            if profile is not None:
                profiles.write_manifest(profile)

    # ---------- presentation ----------
    def transcripts_by_reference(self, ref_ids: list[str]) -> dict[str, list[Transcript]]:
        grouped: dict[str, list[Transcript]] = {rid: [] for rid in ref_ids}
        if ref_ids:
            stmt = select(Transcript).where(Transcript.reference_id.in_(ref_ids))  # type: ignore[attr-defined]
            for tr in self.session.exec(stmt):
                grouped[tr.reference_id].append(tr)
        return grouped

    def read(self, ref: ReferenceAudio, recommended_id: str | None = None,
             transcripts: list[Transcript] | None = None) -> ReferenceRead:
        if transcripts is None:
            transcripts = self.transcripts_by_reference([ref.id])[ref.id]
        profile = self.session.get(VoiceProfile, ref.profile_id) if ref.profile_id else None
        is_primary = profile is not None and profile.primary_reference_id == ref.id
        return self.to_read(ref, recommended_id, transcripts, is_primary)

    @staticmethod
    def to_read(ref: ReferenceAudio, recommended_id: str | None = None,
                transcripts: list[Transcript] | None = None, is_primary: bool = False) -> ReferenceRead:
        ready = ref.status == ReferenceStatus.ANALYZED
        base = f"/api/audio/references/{ref.id}"
        transcripts = transcripts or []
        full = next((t for t in transcripts if t.segment_start_s is None), None)
        has_segment = ref.segment_start_s is not None and ref.segment_end_s is not None
        segment = next((t for t in transcripts if has_segment and t.segment_start_s == ref.segment_start_s
                        and t.segment_end_s == ref.segment_end_s), None)
        estimate = None
        if has_segment and full is not None and full.word_timestamps and full.text == full.asr_text:
            estimate = words_in_range(full.word_timestamps, ref.segment_start_s, ref.segment_end_s) or None
        return ReferenceRead(
            **ref.model_dump(exclude={"sha256", "is_recommended", "updated_at"}),
            quality_label=(ref.analysis or {}).get("quality_label"),
            is_recommended=ref.id == recommended_id,
            is_primary=is_primary,
            urls=ReferenceUrls(original=f"{base}/original", processed=f"{base}/processed" if ready else None,
                               peaks=f"{base}/peaks" if ready else None),
            transcript=transcript_to_read(full) if full else None,
            segment_transcript=transcript_to_read(segment) if segment else None,
            segment_text_estimate=estimate,
        )

    @staticmethod
    def recommended_id(refs: list[ReferenceAudio]) -> str | None:
        """Best heuristic score among analysed references (needs at least two to compare)."""
        scored = [r for r in refs if r.status == ReferenceStatus.ANALYZED and r.quality_score is not None]
        if len(scored) < 2:
            return None
        return max(scored, key=lambda r: r.quality_score or 0).id
