"""Run automatic evaluators on a generation's playable audio and store the results on the Generation row."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import soundfile as sf
from sqlmodel import Session, select

from app.asr.languages import LANGUAGES
from app.asr.manager import ASRManager
from app.audio.storage import AudioStorage
from app.core.errors import AppError, ErrorCode
from app.evaluation.base import EvaluationInput, EvaluatorInfo
from app.evaluation.registry import build_evaluators
from app.generation.markup import MarkupError, TextRun, parse
from app.models.entities import Generation, GenerationSegment, Transcript
from app.models.enums import JobStatus
from app.services.generation_service import GenerationService


def target_text(gen: Generation, segments: list[GenerationSegment]) -> str:
    """The words the engine was asked to say (markup tags removed)."""
    if segments:
        return " ".join(s.text for s in segments)
    if (gen.expression or {}).get("markup", True):
        try:
            return " ".join(" ".join(e.text for e in parse(gen.text).events if isinstance(e, TextRun)).split())
        except MarkupError:
            pass
    return gen.text


# Engine language parameters use English names (Qwen3-TTS); the ASR expects ISO codes.
ENGINE_LANGUAGES = {"spanish": "es", "english": "en", "chinese": "zh", "japanese": "ja", "korean": "ko",
                    "german": "de", "french": "fr", "russian": "ru", "portuguese": "pt", "italian": "it"}


def target_language(session: Session, gen: Generation) -> tuple[str | None, str]:
    """Language to force on the ASR, so an accented result is not auto-detected (and translated) as another one.

    Order: the engine's language parameter → the language detected on the reference transcript → auto-detect.
    """
    value = str((gen.params or {}).get("language") or "").strip().lower()
    code = ENGINE_LANGUAGES.get(value, value if value in LANGUAGES else None)
    if code:
        return code, "engine"
    ref_id = (gen.reference_snapshot or {}).get("reference_id")
    if ref_id:
        stmt = select(Transcript.language).where(Transcript.reference_id == ref_id,
                                                 Transcript.language != None)  # noqa: E711
        found = next(iter(session.exec(stmt)), None)
        if found and found in LANGUAGES:
            return found, "reference"
    return None, "auto"


class EvaluationService:
    def __init__(self, session: Session, generations: GenerationService, asr: ASRManager) -> None:
        self.session = session
        self.generations = generations
        self.evaluators = build_evaluators(asr)

    def _reference_audio(self, gen: Generation) -> tuple[np.ndarray | None, int | None]:
        """The reference range the engine cloned from (processed 24 kHz WAV), or (None, None) if there was none."""
        snap = gen.reference_snapshot or {}
        sha = snap.get("sha256")
        if not sha:
            return None, None
        path = AudioStorage(self.generations.settings.data_dir).processed_path(sha)
        if not path.exists():
            return None, None
        info = sf.info(path)
        start = int((snap.get("start_s") or 0) * info.samplerate)
        stop = int(snap["end_s"] * info.samplerate) if snap.get("end_s") is not None else None
        audio, sr = sf.read(path, start=start, stop=stop, dtype="float32", always_2d=False)
        return np.asarray(audio).reshape(-1), sr

    def available(self) -> list[EvaluatorInfo]:
        return [e.info() for e in self.evaluators]

    def evaluate(self, generation_id: str) -> Generation:
        gen = self.generations.get(generation_id)
        if gen.status != JobStatus.COMPLETED:
            raise AppError(ErrorCode.GENERATION_NOT_READY, status_code=409)
        infos = self.available()
        runnable = [e for e, info in zip(self.evaluators, infos, strict=True) if info.available]
        if not runnable:
            raise AppError(ErrorCode.EVALUATION_UNAVAILABLE, status_code=409,
                           details={i.id: i.reason for i in infos})
        audio, sr = sf.read(self.generations.audio_path(gen), dtype="float32", always_2d=False)
        language, language_source = target_language(self.session, gen)
        ref_audio, ref_sr = self._reference_audio(gen)
        data = EvaluationInput(audio=np.asarray(audio).reshape(-1), sample_rate=sr, language=language,
                               target_text=target_text(gen, self.generations.segments_of(gen.id)),
                               reference_audio=ref_audio, reference_sample_rate=ref_sr)
        metrics, details = [], {}
        for evaluator in runnable:
            out = evaluator.evaluate(data)
            metrics += [m.model_dump() for m in out.metrics]
            details[evaluator.id] = out.details
        gen.evaluation = {
            "metrics": metrics,
            "details": details,
            "target_text": data.target_text,
            "language": language,
            "language_source": language_source,
            "unavailable": [i.model_dump() for i in infos if not i.available],
            "output_path": gen.output_path,  # to flag results made on audio that was re-mastered later
            "evaluated_at": datetime.now(UTC).isoformat(),
        }
        self.session.add(gen)
        self.session.commit()
        self.session.refresh(gen)
        return gen
