"""Available evaluators. Planned metrics are listed as unavailable (with the reason) rather than simulated."""

from __future__ import annotations

from app.asr.manager import ASRManager
from app.evaluation.base import EvaluationInput, EvaluationOutput, Evaluator, EvaluatorInfo
from app.evaluation.intelligibility import IntelligibilityEvaluator
from app.evaluation.speaker import SpeakerSimilarityEvaluator, get_speaker_encoder


class PlannedEvaluator:
    def __init__(self, id: str, label: str, description: str, reason: str) -> None:
        self.id, self.label, self.description, self.reason = id, label, description, reason

    def info(self) -> EvaluatorInfo:
        return EvaluatorInfo(id=self.id, label=self.label, description=self.description, available=False,
                             reason=self.reason)

    def evaluate(self, data: EvaluationInput) -> EvaluationOutput:  # pragma: no cover - never called
        raise NotImplementedError


def build_evaluators(asr: ASRManager) -> list[Evaluator]:
    return [
        IntelligibilityEvaluator(asr),
        SpeakerSimilarityEvaluator(get_speaker_encoder()),
        PlannedEvaluator(
            "naturalness", "Naturalidad (MOS estimado)",
            "Predicción automática de la puntuación de oyentes; no sustituye una escucha real.",
            "Requiere un predictor de MOS (p. ej. UTMOS) que no está instalado."),
    ]
