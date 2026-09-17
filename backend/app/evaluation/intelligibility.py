"""Intelligibility estimate: transcribe the generated audio with the local ASR and compare it with the target text.

WER/CER measure how closely the ASR recovers the text. They depend on the ASR itself (errors on names, numbers or
languages it handles poorly) and say nothing about timbre or naturalness.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
from scipy.signal import resample_poly

from app.asr.manager import ASRManager
from app.core.errors import AppError, ErrorCode
from app.evaluation.base import EvaluationInput, EvaluationOutput, EvaluatorInfo, Metric

ASR_SAMPLE_RATE = 16_000
_NUMBER_SEP = re.compile(r"(?<=\d)[.,](?=\d)")
_NON_WORD = re.compile(r"[^\w\s']|_", re.UNICODE)


def normalize(text: str) -> str:
    """Lowercase, NFC, no punctuation (keeps accents and apostrophes inside words), single spaces."""
    text = unicodedata.normalize("NFC", text).lower().replace("’", "'")
    text = _NUMBER_SEP.sub("", text)
    text = _NON_WORD.sub(" ", text)
    return " ".join(w.strip("'") for w in text.split() if w.strip("'"))


def edit_distance(ref: list[str] | str, hyp: list[str] | str) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def error_rates(reference: str, hypothesis: str, language: str | None = None) -> tuple[float, float]:
    """With a supported language, numbers/abbreviations are written out on both sides ("100%" == "cien por ciento")."""
    from app.text.normalization import SUPPORTED_LANGUAGES, normalize_text

    if language in SUPPORTED_LANGUAGES:
        reference = normalize_text(reference, language)[0]
        hypothesis = normalize_text(hypothesis, language)[0]
    ref_n, hyp_n = normalize(reference), normalize(hypothesis)
    ref_words, hyp_words = ref_n.split(), hyp_n.split()
    ref_chars, hyp_chars = ref_n.replace(" ", ""), hyp_n.replace(" ", "")
    wer = edit_distance(ref_words, hyp_words) / max(1, len(ref_words))
    cer = edit_distance(ref_chars, hyp_chars) / max(1, len(ref_chars))
    return wer, cer


class IntelligibilityEvaluator:
    id = "intelligibility"
    label = "Inteligibilidad (ASR)"
    description = ("Transcribe el audio generado con Whisper y lo compara con el texto pedido (WER/CER). "
                   "Estimación automática: depende también de los errores del propio reconocedor.")

    def __init__(self, asr: ASRManager) -> None:
        self.asr = asr

    def info(self) -> EvaluatorInfo:
        reason = None
        if not self.asr.backend.package_available():
            reason = "Falta instalar faster-whisper."
        elif not self.asr.backend.model_installed():
            reason = "El modelo de transcripción no está descargado (Configuración)."
        return EvaluatorInfo(id=self.id, label=self.label, description=self.description, available=reason is None,
                             reason=reason)

    def evaluate(self, data: EvaluationInput) -> EvaluationOutput:
        if not normalize(data.target_text):
            raise AppError(ErrorCode.VALIDATION_ERROR, status_code=422, message="No hay texto con el que comparar.")
        audio = np.asarray(data.audio, dtype=np.float32).reshape(-1)
        if data.sample_rate != ASR_SAMPLE_RATE:
            from math import gcd

            g = gcd(ASR_SAMPLE_RATE, data.sample_rate)
            audio = resample_poly(audio, ASR_SAMPLE_RATE // g, data.sample_rate // g).astype(np.float32)
        result = self.asr.transcribe(audio, data.language)
        wer, cer = error_rates(data.target_text, result.text, data.language or result.language)
        return EvaluationOutput(
            metrics=[
                Metric(id="wer", label="Error de palabras (WER)", value=round(wer, 4), display=f"{wer * 100:.1f} %",
                       better="lower", description="Palabras distintas del texto pedido según el reconocedor."),
                Metric(id="cer", label="Error de caracteres (CER)", value=round(cer, 4), display=f"{cer * 100:.1f} %",
                       better="lower", description="Menos sensible que el WER a pequeñas variaciones de escritura."),
            ],
            details={"transcript": result.text.strip(), "language": result.language, "asr_model": result.model},
        )
