from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from sqlmodel import Session, select
from starlette.concurrency import run_in_threadpool

from app.asr.base import ASR_SAMPLE_RATE, TranscriptionResult
from app.asr.languages import LANGUAGES
from app.asr.manager import ASRManager
from app.audio.storage import AudioStorage
from app.core.config import Settings
from app.core.errors import AppError, ErrorCode
from app.models.entities import ReferenceAudio, Transcript
from app.models.enums import ReferenceStatus, TranscriptSource
from app.schemas.transcription import TranscriptConfidence, TranscriptRead, TranscriptWord

logger = logging.getLogger("voicelab.transcription")

CACHE_VERSION = 1
LOW_LOGPROB = -0.8
HIGH_NO_SPEECH = 0.6
LOW_LANGUAGE_PROB = 0.7
LOW_WORD_PROB = 0.6


def words_in_range(words: list[dict], start_s: float, end_s: float) -> str:
    """Words whose midpoint falls inside [start, end] — an estimate of the text spoken in a segment."""
    picked = [w["word"] for w in words if start_s <= (w["start"] + w["end"]) / 2 <= end_s]
    return "".join(picked).strip()


def _confidence(tr: Transcript) -> tuple[TranscriptConfidence, list[str]]:
    meta = tr.asr_metadata or {}
    conf = TranscriptConfidence(**{k: meta.get(k) for k in TranscriptConfidence.model_fields})
    warnings: list[str] = []
    if tr.source == TranscriptSource.MANUAL:
        return conf, warnings
    if conf.avg_logprob is not None and conf.avg_logprob < LOW_LOGPROB:
        warnings.append("Confianza baja del reconocimiento: revisa la transcripción.")
    if conf.mean_word_probability is not None and conf.mean_word_probability < LOW_WORD_PROB:
        warnings.append("Varias palabras tienen baja probabilidad: pueden estar mal reconocidas.")
    if conf.max_no_speech_prob is not None and conf.max_no_speech_prob > HIGH_NO_SPEECH:
        warnings.append("Parte del audio podría no contener voz.")
    if meta.get("language_requested") is None and conf.language_probability is not None \
            and conf.language_probability < LOW_LANGUAGE_PROB:
        warnings.append("Idioma detectado con poca seguridad: selecciónalo manualmente.")
    if not tr.text.strip():
        warnings.append("No se reconoció texto.")
    return conf, warnings


def to_read(tr: Transcript, include_words: bool = False) -> TranscriptRead:
    conf, warnings = _confidence(tr)
    return TranscriptRead(
        id=tr.id, reference_id=tr.reference_id, text=tr.text, language=tr.language, source=tr.source,
        edited=tr.asr_text is not None and tr.text != tr.asr_text, asr_model=tr.asr_model,
        segment_start_s=tr.segment_start_s, segment_end_s=tr.segment_end_s, confidence=conf,
        warnings=warnings, updated_at=tr.updated_at,
        words=[TranscriptWord(**w) for w in (tr.word_timestamps or [])] if include_words else None,
    )


class TranscriptionService:
    def __init__(self, session: Session, settings: Settings, asr: ASRManager) -> None:
        self.session = session
        self.settings = settings
        self.asr = asr
        self.storage = AudioStorage(settings.data_dir)
        self.cache_dir = settings.data_dir / "cache" / "transcripts"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ----- queries -----
    def _reference(self, reference_id: str) -> ReferenceAudio:
        ref = self.session.get(ReferenceAudio, reference_id)
        if ref is None:
            raise AppError(ErrorCode.NOT_FOUND, status_code=404, message="La referencia no existe.")
        return ref

    def list_for(self, reference_id: str) -> list[Transcript]:
        self._reference(reference_id)
        stmt = select(Transcript).where(Transcript.reference_id == reference_id).order_by(Transcript.created_at)
        return list(self.session.exec(stmt))

    def get(self, transcript_id: str) -> Transcript:
        tr = self.session.get(Transcript, transcript_id)
        if tr is None:
            raise AppError(ErrorCode.NOT_FOUND, status_code=404, message="La transcripción no existe.")
        return tr

    # ----- commands -----
    async def transcribe(self, reference_id: str, scope: str, language: str | None, force: bool) -> Transcript:
        ref = self._reference(reference_id)
        if ref.status != ReferenceStatus.ANALYZED:
            raise AppError(ErrorCode.VALIDATION_ERROR, status_code=422,
                           message="La referencia debe estar analizada antes de transcribirla.")
        language = (language or "").strip().lower() or None
        if language is not None and language not in LANGUAGES:
            raise AppError(ErrorCode.VALIDATION_ERROR, status_code=422, message="Idioma no admitido.",
                           details={"idioma": language})
        start = end = None
        if scope == "segment":
            if ref.segment_start_s is None or ref.segment_end_s is None:
                raise AppError(ErrorCode.VALIDATION_ERROR, status_code=422,
                               message="Guarda primero un segmento de referencia para transcribirlo.")
            start, end = ref.segment_start_s, ref.segment_end_s

        cache_file = self._cache_path(ref.sha256, language, start, end)
        result = None if force else self._read_cache(cache_file)
        if result is None:
            audio = await run_in_threadpool(self._load_audio, self.storage.processed_path(ref.sha256), start, end)
            result = await run_in_threadpool(self.asr.transcribe, audio, language)
            cache_file.write_text(result.model_dump_json(), encoding="utf-8")
        return self._upsert(ref, result, language, start, end)

    def update_text(self, transcript_id: str, text: str) -> Transcript:
        tr = self.get(transcript_id)
        if tr.asr_text is None:
            tr.asr_text = tr.text
        tr.text = text.strip()
        tr.source = TranscriptSource.MANUAL if tr.text != tr.asr_text else TranscriptSource.ASR
        return self._save(tr)

    def revert(self, transcript_id: str) -> Transcript:
        tr = self.get(transcript_id)
        if tr.asr_text is not None:
            tr.text, tr.source = tr.asr_text, TranscriptSource.ASR
        return self._save(tr)

    # ----- helpers -----
    def _save(self, tr: Transcript) -> Transcript:
        tr.updated_at = datetime.now(UTC)
        self.session.add(tr)
        self.session.commit()
        self.session.refresh(tr)
        return tr

    def _cache_path(self, sha: str, language: str | None, start: float | None, end: float | None) -> Path:
        key = f"v{CACHE_VERSION}|{sha}|{self.asr.backend.model_name}|{language or 'auto'}|{start}|{end}"
        return self.cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    @staticmethod
    def _read_cache(path: Path) -> TranscriptionResult | None:
        if not path.exists():
            return None
        try:
            return TranscriptionResult.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            return None

    @staticmethod
    def _load_audio(path: Path, start: float | None, end: float | None) -> np.ndarray:
        if not path.exists():
            raise AppError(ErrorCode.NOT_FOUND, status_code=404, message="El audio procesado no está disponible.")
        info = sf.info(path)
        frames = (int(start * info.samplerate), int(end * info.samplerate)) if start is not None else (0, -1)
        samples, sr = sf.read(path, start=frames[0], stop=None if frames[1] == -1 else frames[1],
                              dtype="float32", always_2d=False)
        if sr == 24_000:  # internal rate → Whisper's 16 kHz, exact 2/3 ratio
            samples = resample_poly(samples, 2, 3)
        elif sr != ASR_SAMPLE_RATE:
            from math import gcd

            g = gcd(ASR_SAMPLE_RATE, sr)
            samples = resample_poly(samples, ASR_SAMPLE_RATE // g, sr // g)
        return samples.astype(np.float32, copy=False)

    def _upsert(self, ref: ReferenceAudio, result: TranscriptionResult, language: str | None,
                start: float | None, end: float | None) -> Transcript:
        stmt = select(Transcript).where(Transcript.reference_id == ref.id,
                                        Transcript.segment_start_s == start, Transcript.segment_end_s == end)
        tr = self.session.exec(stmt).first() or Transcript(reference_id=ref.id, text="")
        offset = start or 0.0
        words = [{**w.model_dump(), "start": round(w.start + offset, 3), "end": round(w.end + offset, 3)}
                 for w in result.words]
        seg_count = len(result.segments)
        tr.text = tr.asr_text = result.text
        tr.source = TranscriptSource.ASR
        tr.language = result.language
        tr.asr_model = result.model
        tr.segment_start_s, tr.segment_end_s = start, end
        tr.word_timestamps = words
        tr.asr_metadata = {
            "language_requested": language,
            "language_probability": result.language_probability,
            "avg_logprob": round(sum(s.avg_logprob for s in result.segments) / seg_count, 4) if seg_count else None,
            "max_no_speech_prob": round(max(s.no_speech_prob for s in result.segments), 4) if seg_count else None,
            "mean_word_probability": round(sum(w.probability for w in result.words) / len(result.words), 4)
            if result.words else None,
            "device": result.device,
            "compute_type": result.compute_type,
            "elapsed_s": result.elapsed_s,
        }
        logger.info("reference_transcribed", extra={"reference_id": ref.id, "scope": "segment" if start is not None
                                                    else "full", "language": result.language})
        return self._save(tr)
