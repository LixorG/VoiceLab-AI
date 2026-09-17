"""Pronunciation dictionary (global or per voice profile) and text normalisation before generation."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlmodel import Session, or_, select

from app.core.errors import AppError, ErrorCode
from app.models.entities import PronunciationEntry, VoiceProfile
from app.schemas.pronunciation import NormalizePreview, NormalizePreviewRead, PronunciationInput, TextChangeRead
from app.text.normalization import SUPPORTED_LANGUAGES, DictionaryEntry, normalize_text, unique_changes

# Engine language parameters use English names (Qwen3-TTS).
ENGINE_LANGUAGES = {"spanish": "es", "english": "en", "chinese": "zh", "japanese": "ja", "korean": "ko",
                    "german": "de", "french": "fr", "russian": "ru", "portuguese": "pt", "italian": "it"}


def language_code(value: object) -> str | None:
    """'Spanish' / 'es' / 'es-CO' → 'es'; 'Auto' or empty → None."""
    text = str(value or "").strip().lower()
    if not text or text == "auto":
        return None
    if text in ENGINE_LANGUAGES:
        return ENGINE_LANGUAGES[text]
    return text[:2] if len(text) == 2 or (len(text) == 5 and text[2] in "-_") else None


class PronunciationService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def list(self, profile_id: str | None = None, include_global: bool = True) -> list[PronunciationEntry]:
        stmt = select(PronunciationEntry)
        if profile_id is not None:
            cond = PronunciationEntry.profile_id == profile_id
            glob = PronunciationEntry.profile_id == None  # noqa: E711
            stmt = stmt.where(or_(cond, glob) if include_global else cond)
        return list(self.session.exec(stmt.order_by(PronunciationEntry.term)))

    def get(self, entry_id: str) -> PronunciationEntry:
        entry = self.session.get(PronunciationEntry, entry_id)
        if entry is None:
            raise AppError(ErrorCode.PRONUNCIATION_NOT_FOUND, status_code=404)
        return entry

    def _check_profile(self, profile_id: str | None) -> None:
        if profile_id is not None and self.session.get(VoiceProfile, profile_id) is None:
            raise AppError(ErrorCode.NOT_FOUND, status_code=404, message="El perfil de voz no existe.")

    def create(self, body: PronunciationInput) -> PronunciationEntry:
        self._check_profile(body.profile_id)
        entry = PronunciationEntry(**body.model_dump())
        self.session.add(entry)
        self.session.commit()
        self.session.refresh(entry)
        return entry

    def update(self, entry_id: str, body: PronunciationInput) -> PronunciationEntry:
        entry = self.get(entry_id)
        self._check_profile(body.profile_id)
        for key, value in body.model_dump().items():
            setattr(entry, key, value)
        entry.updated_at = datetime.now(UTC)
        self.session.add(entry)
        self.session.commit()
        self.session.refresh(entry)
        return entry

    def delete(self, entry_id: str) -> None:
        self.session.delete(self.get(entry_id))
        self.session.commit()

    def dictionary(self, profile_id: str | None) -> list[DictionaryEntry]:
        """Global entries plus the profile's; a profile entry wins over a global one with the same term."""
        entries = self.list(profile_id) if profile_id else self.list()
        by_term: dict[str, PronunciationEntry] = {}
        for entry in sorted(entries, key=lambda e: e.profile_id is not None):
            if entry.profile_id in (None, profile_id):
                by_term[entry.term.lower()] = entry
        return [DictionaryEntry(e.term, e.replacement, e.case_sensitive) for e in by_term.values()]

    def resolve_language(self, params: dict, profile_id: str | None) -> str | None:
        """Engine language parameter → profile language → Spanish (the app's default content language)."""
        code = language_code(params.get("language"))
        if code:
            return code
        profile = self.session.get(VoiceProfile, profile_id) if profile_id else None
        if profile is not None and language_code(profile.language):
            return language_code(profile.language)
        return "es"

    def preview(self, body: NormalizePreview) -> NormalizePreviewRead:
        language = language_code(body.language)
        text, changes = normalize_text(body.text, language, self.dictionary(body.profile_id), numbers=body.numbers)
        return NormalizePreviewRead(
            text=text, language=language, numbers_supported=language in SUPPORTED_LANGUAGES,
            changes=[TextChangeRead(original=c.original, replacement=c.replacement, kind=c.kind)
                     for c in unique_changes(changes)])
