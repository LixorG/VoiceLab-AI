"""Pronunciation dictionary (global or per voice profile) and text normalisation before generation."""

from __future__ import annotations

import re
import unicodedata
from datetime import UTC, datetime

from sqlmodel import Session, or_, select

from app.core.errors import AppError, ErrorCode
from app.models.entities import PronunciationEntry, VoiceProfile
from app.schemas.pronunciation import NormalizePreview, NormalizePreviewRead, PronunciationInput, TextChangeRead
from app.text.normalization import SUPPORTED_LANGUAGES, DictionaryEntry, normalize_text, unique_changes

# Engine language parameters use English names (Qwen3-TTS); voice profiles have a free field whose placeholder
# suggests Spanish names ("Español"), so both are understood (compared without accents).
ENGINE_LANGUAGES = {"spanish": "es", "english": "en", "chinese": "zh", "japanese": "ja", "korean": "ko",
                    "german": "de", "french": "fr", "russian": "ru", "portuguese": "pt", "italian": "it",
                    "espanol": "es", "castellano": "es", "ingles": "en", "chino": "zh", "japones": "ja",
                    "coreano": "ko", "aleman": "de", "frances": "fr", "ruso": "ru", "portugues": "pt",
                    "italiano": "it"}
# Frequent function words, to guess the language of a text when neither the engine nor the voice says it.
_STOPWORDS = {
    "es": frozenset("el la los las de del que y en un una por con para es no se su al lo como más pero sus le ya "
                    "o este esta sí porque muy sin sobre también me hay tu te yo mi".split()),
    "en": frozenset("the and of to a in is you that it for on with are this be your not as at have was but they "
                    "i my me we he she do if so what".split()),
}
_WORDS = re.compile(r"[^\W\d_]+", re.UNICODE)


def _plain(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def language_code(value: object) -> str | None:
    """'Spanish' / 'Inglés' / 'es' / 'es-CO' → code; 'Auto', empty or unknown → None."""
    text = _plain(str(value or "")).strip().lower()
    if not text or text == "auto":
        return None
    if text in ENGINE_LANGUAGES:
        return ENGINE_LANGUAGES[text]
    first = text.split()[0].strip("()")  # "Inglés (EE. UU.)"
    if first in ENGINE_LANGUAGES:
        return ENGINE_LANGUAGES[first]
    return text[:2] if len(text) == 2 or (len(text) > 3 and text[2] in "-_" and text[:2].isalpha()) else None


def guess_language(text: str) -> str | None:
    """'es' or 'en' when the text clearly uses one language's function words; None when unsure."""
    words = [w.lower() for w in _WORDS.findall(text)]
    counts = {lang: sum(w in stop for w in words) for lang, stop in _STOPWORDS.items()}
    best = max(counts, key=lambda lang: counts[lang])
    others = max(v for lang, v in counts.items() if lang != best)
    return best if counts[best] >= 2 and counts[best] >= 2 * others else None


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

    def resolve_language(self, params: dict, profile_id: str | None, text: str = "") -> str | None:
        """Engine language parameter → profile language → the text's own language → Spanish (app default)."""
        code = language_code(params.get("language"))
        if code:
            return code
        profile = self.session.get(VoiceProfile, profile_id) if profile_id else None
        if profile is not None and language_code(profile.language):
            return language_code(profile.language)
        return guess_language(text) or "es"

    def preview(self, body: NormalizePreview) -> NormalizePreviewRead:
        language = language_code(body.language)
        text, changes = normalize_text(body.text, language, self.dictionary(body.profile_id), numbers=body.numbers)
        return NormalizePreviewRead(
            text=text, language=language, numbers_supported=language in SUPPORTED_LANGUAGES,
            changes=[TextChangeRead(original=c.original, replacement=c.replacement, kind=c.kind)
                     for c in unique_changes(changes)])
