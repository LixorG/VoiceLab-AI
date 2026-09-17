"""Voice profiles: CRUD, primary/recommended reference, recommended engine settings, manifest, export/import."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import unicodedata
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlmodel import Session, select

from app.audio.validation import ALLOWED_FORMATS, sanitize_filename
from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.engines.registry import EngineRegistry
from app.models.entities import Generation, ReferenceAudio, Transcript, VoiceProfile
from app.models.enums import ReferenceStatus, TranscriptSource
from app.schemas.profiles import (
    ProfileCreate,
    ProfileRead,
    ProfileStats,
    ProfileUpdate,
    RecommendedSetting,
    RecommendedSettingUpdate,
)
from app.services.reference_service import ReferenceService

logger = logging.getLogger("voicelab.profiles")

EXPORT_FORMAT = "voicelab.voiceprofile"
EXPORT_VERSION = 1
MAX_IMPORT_ENTRIES = 250
MAX_MANIFEST_BYTES = 5 * 1024 * 1024
_REFERENCE_ENTRY = re.compile(r"^references/([0-9a-f]{64})\.(" + "|".join(ALLOWED_FORMATS) + r")$")


def slugify(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    return slug[:60] or "voz"


def quality_label(score: float | None) -> str | None:
    if score is None:
        return None
    return "Excelente" if score >= 85 else "Buena" if score >= 70 else "Aceptable" if score >= 50 else "Baja"


class ProfileService:
    def __init__(self, session: Session, settings: Settings, registry: EngineRegistry) -> None:
        self.session = session
        self.settings = settings
        self.registry = registry
        self.references = ReferenceService(session, settings)

    # ------------------------------------------------------------------ queries
    def get(self, profile_id: str) -> VoiceProfile:
        profile = self.session.get(VoiceProfile, profile_id)
        if profile is None:
            raise AppError(ErrorCode.PROFILE_NOT_FOUND, status_code=404)
        return profile

    def list(self) -> list[VoiceProfile]:
        return list(self.session.exec(select(VoiceProfile).order_by(VoiceProfile.name)))

    def refs_of(self, profile_id: str) -> list[ReferenceAudio]:
        return self.references.list(profile_id)

    def to_read(self, profile: VoiceProfile) -> ProfileRead:
        refs = self.refs_of(profile.id)
        analyzed = [r for r in refs if r.status == ReferenceStatus.ANALYZED]
        transcripts = self.references.transcripts_by_reference([r.id for r in refs])
        scores = [r.quality_score for r in analyzed if r.quality_score is not None]
        avg = round(sum(scores) / len(scores), 1) if scores else None
        stats = ProfileStats(
            reference_count=len(refs),
            analyzed_count=len(analyzed),
            transcribed_count=sum(1 for r in refs if any(t.segment_start_s is None for t in transcripts[r.id])),
            total_duration_s=round(sum(r.duration_s or 0 for r in refs), 2),
            total_speech_s=round(sum((r.analysis or {}).get("effective_speech_s", 0) for r in analyzed), 2),
            average_quality=avg,
            quality_label=quality_label(avg),
            emotions=sorted({r.emotion_tag for r in refs if r.emotion_tag}),
        )
        settings = {k: RecommendedSetting.model_validate(v) for k, v in (profile.recommended_settings or {}).items()}
        return ProfileRead(
            id=profile.id, name=profile.name, slug=profile.slug, description=profile.description,
            language=profile.language, default_engine=profile.default_engine,
            primary_reference_id=profile.primary_reference_id,
            recommended_reference_id=self.references.recommended_id(refs), recommended_settings=settings,
            stats=stats, created_at=profile.created_at, updated_at=profile.updated_at,
        )

    def generation_reference_id(self, profile_id: str) -> str | None:
        """Reference used when generating with a profile: primary, else recommended, else first analysed."""
        profile = self.get(profile_id)
        refs = [r for r in self.refs_of(profile.id) if r.status == ReferenceStatus.ANALYZED]
        if profile.primary_reference_id and any(r.id == profile.primary_reference_id for r in refs):
            return profile.primary_reference_id
        return self.references.recommended_id(refs) or (refs[0].id if refs else None)

    # ------------------------------------------------------------------ commands
    def create(self, data: ProfileCreate) -> VoiceProfile:
        self._check_engine(data.default_engine)
        profile = VoiceProfile(name=data.name, slug=self._unique_slug(slugify(data.name)),
                               description=data.description, language=data.language,
                               default_engine=data.default_engine, recommended_settings={})
        return self._save(profile)

    def update(self, profile_id: str, data: ProfileUpdate) -> VoiceProfile:
        profile = self.get(profile_id)
        fields = data.model_dump(exclude_unset=True)
        if "default_engine" in fields:
            self._check_engine(fields["default_engine"])
        if fields.get("primary_reference_id"):
            ref = self.session.get(ReferenceAudio, fields["primary_reference_id"])
            if ref is None or ref.profile_id != profile.id:
                raise AppError(ErrorCode.VALIDATION_ERROR, status_code=422,
                               message="La referencia principal debe pertenecer a este perfil.")
        if "name" in fields and fields["name"] is not None:
            fields["name"] = fields["name"].strip()
            if not fields["name"]:
                raise AppError(ErrorCode.VALIDATION_ERROR, status_code=422, message="El nombre no puede estar vacío.")
        for key, value in fields.items():
            setattr(profile, key, value)
        return self._save(profile)

    def set_recommended(self, profile_id: str, engine_id: str, data: RecommendedSettingUpdate) -> VoiceProfile:
        profile = self.get(profile_id)
        engine = self.registry.get(engine_id)
        variant = engine.variant(data.variant).id
        params = engine.validate_parameters(data.params, variant)  # only real, valid parameters are stored
        settings = dict(profile.recommended_settings or {})
        settings[engine.id] = RecommendedSetting(variant=variant, params=params, note=data.note,
                                                 updated_at=datetime.now(UTC)).model_dump(mode="json")
        profile.recommended_settings = settings
        return self._save(profile)

    def remove_recommended(self, profile_id: str, engine_id: str) -> VoiceProfile:
        profile = self.get(profile_id)
        settings = dict(profile.recommended_settings or {})
        settings.pop(engine_id, None)
        profile.recommended_settings = settings
        return self._save(profile)

    def delete(self, profile_id: str, delete_references: bool) -> None:
        profile = self.get(profile_id)
        for ref in self.refs_of(profile.id):
            if delete_references:
                self.references.delete(ref.id)
            else:
                ref.profile_id = None
                self.session.add(ref)
        for gen in self.session.exec(select(Generation).where(Generation.profile_id == profile.id)):
            gen.profile_id = None
            self.session.add(gen)
        self.session.flush()
        self.session.delete(profile)
        self.session.commit()
        manifest = self._manifest_dir(profile) / "profile.json"
        manifest.unlink(missing_ok=True)
        try:
            manifest.parent.rmdir()
        except OSError:
            pass

    # ------------------------------------------------------------------ export / import
    def export_manifest(self, profile: VoiceProfile) -> dict[str, Any]:
        refs = self.refs_of(profile.id)
        transcripts = self.references.transcripts_by_reference([r.id for r in refs])
        return {
            "format": EXPORT_FORMAT,
            "version": EXPORT_VERSION,
            "exported_at": datetime.now(UTC).isoformat(),
            "profile": {
                "name": profile.name, "slug": profile.slug, "description": profile.description,
                "language": profile.language, "default_engine": profile.default_engine,
                "recommended_settings": profile.recommended_settings or {},
            },
            "references": [
                {
                    "sha256": r.sha256, "file": f"references/{r.sha256}.{r.format}", "original_name": r.original_name,
                    "format": r.format, "emotion_tag": r.emotion_tag, "segment_start_s": r.segment_start_s,
                    "segment_end_s": r.segment_end_s, "is_primary": r.id == profile.primary_reference_id,
                    "duration_s": r.duration_s, "quality_score": r.quality_score,
                    "transcripts": [
                        {"text": t.text, "asr_text": t.asr_text, "language": t.language, "source": t.source.value,
                         "asr_model": t.asr_model, "segment_start_s": t.segment_start_s,
                         "segment_end_s": t.segment_end_s, "word_timestamps": t.word_timestamps}
                        for t in transcripts[r.id]
                    ],
                }
                for r in refs
            ],
        }

    def export_zip(self, profile_id: str) -> tuple[bytes, str]:
        profile = self.get(profile_id)
        manifest = self.export_manifest(profile)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("profile.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            for entry in manifest["references"]:
                original = self.references.storage.find_original(entry["sha256"])
                if original is not None and entry["file"] not in zf.namelist():
                    zf.write(original, entry["file"])
        return buffer.getvalue(), f"{profile.slug}.voiceprofile"

    async def import_zip(self, data: bytes) -> VoiceProfile:
        if len(data) > self.settings.max_upload_bytes:
            raise AppError(ErrorCode.FILE_TOO_LARGE, status_code=413)
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise AppError(ErrorCode.PROFILE_IMPORT_ERROR, status_code=422) from exc

        with zf:
            infos = zf.infolist()
            if len(infos) > MAX_IMPORT_ENTRIES:
                raise AppError(ErrorCode.PROFILE_IMPORT_ERROR, status_code=422,
                               message="El archivo contiene demasiadas entradas.")
            names = {i.filename: i for i in infos if not i.is_dir()}
            for name in names:  # strict whitelist: no absolute paths, no traversal, no unknown files
                if name != "profile.json" and not _REFERENCE_ENTRY.match(name):
                    raise AppError(ErrorCode.PROFILE_IMPORT_ERROR, status_code=422,
                                   message="El archivo contiene entradas no permitidas.",
                                   details={"entrada": name[:120]})
            if sum(i.file_size for i in names.values()) > self.settings.max_upload_bytes * 2:
                raise AppError(ErrorCode.FILE_TOO_LARGE, status_code=413)
            if "profile.json" not in names or names["profile.json"].file_size > MAX_MANIFEST_BYTES:
                raise AppError(ErrorCode.PROFILE_IMPORT_ERROR, status_code=422,
                               message="Falta profile.json o es inválido.")
            try:
                manifest = json.loads(zf.read("profile.json").decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                raise AppError(ErrorCode.PROFILE_IMPORT_ERROR, status_code=422) from exc
            if manifest.get("format") != EXPORT_FORMAT or manifest.get("version") != EXPORT_VERSION:
                raise AppError(ErrorCode.PROFILE_IMPORT_ERROR, status_code=422,
                               message="Formato o versión de perfil no compatible.")

            meta = manifest.get("profile") or {}
            name = str(meta.get("name") or "Perfil importado")[:66]
            if self.session.exec(select(VoiceProfile).where(VoiceProfile.name == name)).first() is not None:
                name = f"{name} (importado)"  # keep both profiles distinguishable in the UI
            profile = self.create(ProfileCreate(name=name,
                                                description=meta.get("description"), language=meta.get("language"),
                                                default_engine=meta.get("default_engine")
                                                if meta.get("default_engine") in {e.id for e in self.registry.all()}
                                                else None))
            try:
                await self._import_references(zf, names, manifest.get("references") or [], profile)
                self._import_settings(profile, meta.get("recommended_settings") or {})
            except BaseException:
                self.delete(profile.id, delete_references=True)
                raise
        return self._save(profile)

    async def _import_references(self, zf: zipfile.ZipFile, names: dict, entries: list[dict],
                                 profile: VoiceProfile) -> None:
        for entry in entries:
            file = str(entry.get("file", ""))
            match = _REFERENCE_ENTRY.match(file)
            if not match or file not in names:
                raise AppError(ErrorCode.PROFILE_IMPORT_ERROR, status_code=422,
                               message="Una referencia del perfil no está en el archivo.",
                               details={"entrada": file[:120]})
            payload = zf.read(file)
            sha = hashlib.sha256(payload).hexdigest()
            if sha != match.group(1):
                raise AppError(ErrorCode.PROFILE_IMPORT_ERROR, status_code=422,
                               message="Un audio del perfil está dañado (hash no coincide).")
            tmp = self.references.storage.temp_path(".import")
            tmp.write_bytes(payload)
            name = sanitize_filename(str(entry.get("original_name") or f"{sha[:8]}.{match.group(2)}"))
            if Path(name).suffix.lower().lstrip(".") != match.group(2):
                name = f"{Path(name).stem}.{match.group(2)}"
            ref, _ = await self.references.ingest_temp(tmp, sha, len(payload), name, match.group(2), profile.id)
            self._restore_reference_state(ref, entry)
            if entry.get("is_primary"):
                profile.primary_reference_id = ref.id

    def _restore_reference_state(self, ref: ReferenceAudio, entry: dict) -> None:
        from app.voice_profiles.emotions import EMOTIONS

        if entry.get("emotion_tag") in EMOTIONS:
            ref.emotion_tag = entry["emotion_tag"]
        start, end = entry.get("segment_start_s"), entry.get("segment_end_s")
        numeric = isinstance(start, int | float) and isinstance(end, int | float)
        if numeric and 0 <= start < end <= (ref.duration_s or 0) + 0.05:
            ref.segment_start_s, ref.segment_end_s = float(start), float(end)
        self.session.add(ref)
        for t in entry.get("transcripts") or []:
            text = str(t.get("text") or "").strip()
            if not text:
                continue
            source = TranscriptSource.MANUAL if t.get("source") == "manual" else TranscriptSource.ASR
            words = t.get("word_timestamps") if isinstance(t.get("word_timestamps"), list) else None
            self.session.add(Transcript(
                reference_id=ref.id, text=text[:20_000], asr_text=(t.get("asr_text") or None), source=source,
                language=t.get("language"), asr_model=t.get("asr_model"), word_timestamps=words,
                segment_start_s=t.get("segment_start_s"), segment_end_s=t.get("segment_end_s")))
        self.session.commit()

    def _import_settings(self, profile: VoiceProfile, settings: dict) -> None:
        for engine_id, setting in settings.items():
            try:
                self.set_recommended(profile.id, engine_id, RecommendedSettingUpdate(
                    variant=setting.get("variant"), params=setting.get("params") or {}, note=setting.get("note")))
            except AppError:
                logger.warning("profile_import_setting_skipped", extra={"engine": engine_id})

    # ------------------------------------------------------------------ helpers
    def _check_engine(self, engine_id: str | None) -> None:
        if engine_id is not None:
            self.registry.get(engine_id)

    def _unique_slug(self, base: str) -> str:
        slug, n = base, 2
        while self.session.exec(select(VoiceProfile).where(VoiceProfile.slug == slug)).first() is not None:
            slug = f"{base}-{n}"
            n += 1
        return slug

    def _manifest_dir(self, profile: VoiceProfile) -> Path:
        return self.settings.data_dir / "voices" / profile.slug

    def _save(self, profile: VoiceProfile) -> VoiceProfile:
        profile.updated_at = datetime.now(UTC)
        self.session.add(profile)
        self.session.commit()
        self.session.refresh(profile)
        self.write_manifest(profile)
        return profile

    def write_manifest(self, profile: VoiceProfile) -> None:
        """Human-readable snapshot in data/voices/<slug>/profile.json (audio stays content-addressed)."""
        folder = self._manifest_dir(profile)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "profile.json").write_text(json.dumps(self.export_manifest(profile), ensure_ascii=False, indent=2),
                                             encoding="utf-8")
