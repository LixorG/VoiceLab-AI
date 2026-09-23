"""Generate the same text a few times and keep the take that reads it best.

A model reads the same sentence differently on every seed: most takes are fine and one in a handful skips a word,
mumbles the ending or runs long. With more than one take VoiceLab scores each one with the evaluators the app
already has — Whisper for the words that were actually said, WavLM for how close the voice stays to the reference —
plus checks that need no model at all (a take much longer than its siblings, clipping, or mostly silence).

The score is an automatic estimate, like every other metric here: it catches the clearly bad takes, it does not
judge interpretation. Ties keep the first take, so asking for one take behaves exactly as before.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

WER_WEIGHT, SIMILARITY_WEIGHT = 0.65, 0.35
DURATION_TOLERANCE = 1.35  # longer than this times the median of the takes = probably a loop or a long silence
DURATION_TOLERANCE_ESTIMATE = 1.6  # against the estimate from the text (coarser), give it more room
CHARS_PER_SECOND = 14.0  # the same speech-rate estimate the Qwen adapter uses for its progress
LOW_SPEECH_RATIO = 0.45  # under this fraction of the take is voice: mostly silence
CLIPPING_PEAK = 0.999
FRAME_MS = 20


@dataclass
class TakeScore:
    take: int
    seed: int | None
    duration_s: float
    speech_ratio: float
    peak: float
    wer: float | None = None
    similarity: float | None = None
    score: float = 0.0
    notes: list[str] | None = None  # why it lost points, in Spanish


def speech_ratio(audio: np.ndarray, sample_rate: int) -> float:
    """Fraction of the take with voice (frames above 40 dB under the loudest one)."""
    frame = max(1, sample_rate * FRAME_MS // 1000)
    usable = audio.size // frame * frame
    if usable < frame:
        return 0.0
    rms = np.sqrt(np.mean(audio[:usable].astype(np.float64).reshape(-1, frame) ** 2, axis=1))
    level = 20 * np.log10(rms + 1e-9)
    return float(np.mean(level > max(level.max() - 40, -60)))


def measure(take: int, seed: int | None, audio: np.ndarray, sample_rate: int) -> TakeScore:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    return TakeScore(take=take, seed=seed, duration_s=round(audio.size / max(sample_rate, 1), 3),
                     speech_ratio=round(speech_ratio(audio, sample_rate), 3),
                     peak=round(float(np.max(np.abs(audio))) if audio.size else 0.0, 4), notes=[])


def add_wer(score: TakeScore, audio: np.ndarray, sample_rate: int, text: str, language: str | None,
            evaluator: Any) -> None:
    """Word error rate of this take against the text that was asked for (Whisper)."""
    from app.evaluation.base import EvaluationInput

    result = evaluator.evaluate(EvaluationInput(audio=audio, sample_rate=sample_rate, target_text=text,
                                                language=language))
    score.wer = next((m.value for m in result.metrics if m.id == "wer"), None)


def add_similarity(score: TakeScore, audio: np.ndarray, sample_rate: int, encoder: Any,
                   reference: np.ndarray) -> None:
    from app.evaluation.speaker import cosine

    score.similarity = round(cosine(encoder.embed(audio, sample_rate), reference), 4)


def rank(scores: list[TakeScore], expected_s: float | None = None) -> list[TakeScore]:
    """Fill in `score` and `notes` for every take (higher is better) and return them best first.

    Length is judged against the other takes when there are at least three (their median is robust); with two the
    median sits between them and would punish the shorter one, so the estimate from the text is used instead.
    """
    durations = [s.duration_s for s in scores if s.duration_s > 0]
    median = float(np.median(durations)) if durations else 0.0
    if len(durations) < 3 and expected_s:
        median, tolerance = expected_s, DURATION_TOLERANCE_ESTIMATE
    else:
        tolerance = DURATION_TOLERANCE
    for s in scores:
        notes: list[str] = []
        parts, weights = [], []
        if s.wer is not None:
            parts.append(max(0.0, 1.0 - s.wer) * WER_WEIGHT)
            weights.append(WER_WEIGHT)
            if s.wer > 0.1:
                notes.append(f"{s.wer * 100:.0f} % de palabras distintas del texto")
        if s.similarity is not None:
            parts.append(max(0.0, min(1.0, s.similarity)) * SIMILARITY_WEIGHT)
            weights.append(SIMILARITY_WEIGHT)
        value = sum(parts) / sum(weights) if weights else 0.5  # no evaluators: only the checks below decide
        if median and s.duration_s > median * tolerance:
            value -= 0.3
            notes.append("mucho más larga que las demás tomas (posible bucle o silencio final)")
        elif median and s.duration_s < median / tolerance:
            value -= 0.2
            notes.append("mucho más corta que las demás tomas (puede faltar texto)")
        if s.speech_ratio < LOW_SPEECH_RATIO:
            value -= 0.15
            notes.append("más de la mitad es silencio")
        if s.peak >= CLIPPING_PEAK:
            value -= 0.1
            notes.append("el audio satura")
        s.score = round(value, 4)
        s.notes = notes
    return sorted(scores, key=lambda s: (-s.score, s.take))


def expected_seconds(text: str) -> float:
    """Roughly how long this text should take to read, to spot a take that runs away or gets cut."""
    return max(0.5, len(text.strip()) / CHARS_PER_SECOND)


def summary(scores: list[TakeScore], chosen: TakeScore, segment: int | None = None) -> dict:
    """What the app stores and shows: the winner, why, and the runners-up."""
    data: dict[str, Any] = {"chosen": chosen.take, "takes": [asdict(s) for s in sorted(scores, key=lambda s: s.take)]}
    if segment is not None:
        data["segment"] = segment
    return data


def reason(chosen: TakeScore, scores: list[TakeScore]) -> str | None:
    """One line in Spanish for the take that was kept, when there was something to choose between."""
    if len(scores) < 2:
        return None
    if chosen.wer is not None:
        worst = max(s.wer for s in scores if s.wer is not None)
        if worst > (chosen.wer or 0) + 0.02:
            return (f"Mejor toma de {len(scores)}: lee {chosen.wer * 100:.0f} % de palabras distintas del texto "
                    f"(la peor, {worst * 100:.0f} %).")
    if chosen.similarity is not None:
        worst = min(s.similarity for s in scores if s.similarity is not None)
        if (chosen.similarity - worst) > 0.01:
            return (f"Mejor toma de {len(scores)}: la voz se parece {chosen.similarity:.2f} a la referencia "
                    f"(la peor, {worst:.2f}).")
    if any(s.notes for s in scores if s.take != chosen.take):
        return f"Mejor toma de {len(scores)}: las demás tenían problemas de duración, silencio o saturación."
    return f"Mejor toma de {len(scores)}: todas leían bien el texto, se conserva la primera."
