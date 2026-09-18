"""Speaker similarity: how close the generated voice is to the reference voice, as an automatic estimate.

Both clips are turned into speaker embeddings with `microsoft/wavlm-base-plus-sv` (WavLMForXVector, already
available through `transformers`, no extra package) and compared with cosine similarity. The model card gives 0.86
as its same-speaker threshold on VoxCeleb; that is an orientation, not a verdict: the embedding captures timbre
more than accent, and short or noisy clips lower the score.

The model runs on CPU on purpose: it is small (~95 M parameters, ~1 s per clip) and must not compete for VRAM
with the TTS engine. Its weights are only downloaded when the user asks for them.
"""

from __future__ import annotations

import logging
import threading
from math import gcd
from typing import Literal

import numpy as np
from pydantic import BaseModel
from scipy.signal import resample_poly

from app.core.errors import AppError, ErrorCode
from app.evaluation.base import EvaluationInput, EvaluationOutput, EvaluatorInfo, Metric

logger = logging.getLogger("voicelab.evaluation.speaker")

SPEAKER_MODEL = "microsoft/wavlm-base-plus-sv"
SPEAKER_FILES = ("config.json", "preprocessor_config.json", "pytorch_model.bin")
SPEAKER_DOWNLOAD_MB = 405
SPEAKER_SAMPLE_RATE = 16_000
SAME_SPEAKER_THRESHOLD = 0.86  # from the model card (VoxCeleb1); orientation only
MIN_SECONDS = 1.0

DownloadState = Literal["idle", "downloading", "failed"]


class SpeakerModelStatus(BaseModel):
    model: str
    installed: bool
    loaded: bool
    download_state: DownloadState
    download_error: str | None
    download_size_mb: int
    license: str


def to_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate == SPEAKER_SAMPLE_RATE:
        return audio
    g = gcd(SPEAKER_SAMPLE_RATE, sample_rate)
    return resample_poly(audio, SPEAKER_SAMPLE_RATE // g, sample_rate // g).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


class SpeakerEncoder:
    """Lazy, thread-safe WavLM x-vector encoder on CPU with a background download."""

    def __init__(self) -> None:
        self._model = None
        self._extractor = None
        self._lock = threading.Lock()
        self._download_state: DownloadState = "idle"
        self._download_error: str | None = None

    # ----- availability
    @staticmethod
    def package_available() -> bool:
        try:
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def model_installed(self) -> bool:
        from app.engines.common import hf_files_cached

        return self.package_available() and hf_files_cached(SPEAKER_MODEL, SPEAKER_FILES)

    def status(self) -> SpeakerModelStatus:
        return SpeakerModelStatus(
            model=SPEAKER_MODEL, installed=self._download_state != "downloading" and self.model_installed(),
            loaded=self._model is not None, download_state=self._download_state,
            download_error=self._download_error, download_size_mb=SPEAKER_DOWNLOAD_MB,
            license="MIT (repositorio microsoft/unilm; la ficha del modelo no declara licencia)")

    # ----- download
    def start_download(self) -> SpeakerModelStatus:
        if not self.package_available():
            raise AppError(ErrorCode.MODEL_NOT_INSTALLED, status_code=409,
                           message="Falta el paquete transformers para calcular la similitud de voz.")
        if self._download_state == "downloading":
            return self.status()
        self._download_state, self._download_error = "downloading", None
        threading.Thread(target=self._download, name="speaker-model-download", daemon=True).start()
        return self.status()

    def _download(self) -> None:
        try:
            from huggingface_hub import hf_hub_download

            from app.engines.common import prepare_hf_cache

            prepare_hf_cache()
            for filename in SPEAKER_FILES:
                hf_hub_download(repo_id=SPEAKER_MODEL, filename=filename)
            self._download_state = "idle"
            logger.info("speaker_model_downloaded")
        except Exception as exc:  # noqa: BLE001  (reported to the user as a failed download)
            logger.warning("speaker_model_download_failed", extra={"error": str(exc)})
            self._download_state, self._download_error = "failed", "No se pudo descargar el modelo. Revisa la conexión."

    # ----- inference
    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.model_installed():
            raise AppError(ErrorCode.EVALUATION_UNAVAILABLE, status_code=409,
                           message="El modelo de similitud de voz no está descargado (Configuración).")
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self._extractor = AutoFeatureExtractor.from_pretrained(SPEAKER_MODEL, local_files_only=True)
        model = WavLMForXVector.from_pretrained(SPEAKER_MODEL, local_files_only=True)
        model.eval()
        self._model = model
        logger.info("speaker_model_loaded")

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch

        wav = to_16k(audio, sample_rate)
        with self._lock:
            self._load()
            inputs = self._extractor(wav, sampling_rate=SPEAKER_SAMPLE_RATE, return_tensors="pt", padding=True)
            with torch.inference_mode():
                embedding = self._model(**inputs).embeddings
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return embedding[0].cpu().numpy()

    def unload(self) -> None:
        with self._lock:
            self._model = self._extractor = None


_encoder = SpeakerEncoder()


def get_speaker_encoder() -> SpeakerEncoder:
    return _encoder


class SpeakerSimilarityEvaluator:
    id = "speaker_similarity"
    label = "Similitud de voz"
    description = ("Compara la voz generada con tu grabación de referencia usando embeddings de hablante "
                   "(WavLM-SV). Estimación automática: mide sobre todo el timbre, no el acento.")

    def __init__(self, encoder: SpeakerEncoder) -> None:
        self.encoder = encoder

    def info(self) -> EvaluatorInfo:
        reason = None
        status = self.encoder.status()
        if not self.encoder.package_available():
            reason = "Falta el paquete transformers."
        elif status.download_state == "downloading":
            reason = "Descargando el modelo de similitud de voz…"
        elif not status.installed:
            reason = f"El modelo de similitud de voz no está descargado (~{SPEAKER_DOWNLOAD_MB} MB, Configuración)."
        return EvaluatorInfo(id=self.id, label=self.label, description=self.description, available=reason is None,
                             reason=reason)

    def evaluate(self, data: EvaluationInput) -> EvaluationOutput:
        if data.reference_audio is None or data.reference_sample_rate is None:
            return EvaluationOutput(metrics=[], details={
                "skipped": "Esta generación no usó una grabación de referencia: no hay con qué comparar."})
        ref_seconds = np.asarray(data.reference_audio).size / data.reference_sample_rate
        gen_seconds = np.asarray(data.audio).size / data.sample_rate
        if min(ref_seconds, gen_seconds) < MIN_SECONDS:
            return EvaluationOutput(metrics=[], details={
                "skipped": f"Hace falta al menos {MIN_SECONDS:.0f} s de audio en cada lado para comparar."})

        score = cosine(self.encoder.embed(data.audio, data.sample_rate),
                       self.encoder.embed(data.reference_audio, data.reference_sample_rate))
        verdict = ("por encima" if score >= SAME_SPEAKER_THRESHOLD else "por debajo")
        return EvaluationOutput(
            metrics=[Metric(
                id="speaker_similarity", label="Similitud de voz", value=round(score, 4),
                display=f"{score:.2f}", better="higher",
                description=f"Similitud coseno entre embeddings de hablante (−1 a 1). El modelo usa "
                            f"{SAME_SPEAKER_THRESHOLD:.2f} como umbral orientativo de «misma persona»; este "
                            f"resultado queda {verdict}. Mide timbre más que acento.")],
            details={"model": SPEAKER_MODEL, "threshold": SAME_SPEAKER_THRESHOLD,
                     "reference_seconds": round(ref_seconds, 2), "generated_seconds": round(gen_seconds, 2)})
