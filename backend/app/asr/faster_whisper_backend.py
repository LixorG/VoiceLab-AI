"""faster-whisper (CTranslate2) implementation of ASRBackend."""

from __future__ import annotations

import gc
import importlib.util
import logging
import os
import time
from pathlib import Path

import numpy as np

from app.asr.base import ASRSegment, ASRWord, TranscriptionResult
from app.core.errors import AppError, ErrorCode

logger = logging.getLogger("voicelab.asr")

_dll_dirs_added = False


def _add_cuda_dll_directories() -> None:
    """On Windows, CTranslate2 needs cuBLAS/cuDNN DLLs; reuse the ones shipped with PyTorch wheels."""
    global _dll_dirs_added
    if _dll_dirs_added or os.name != "nt":
        return
    for package in ("torch", "nvidia"):
        spec = importlib.util.find_spec(package)
        for location in (spec.submodule_search_locations or []) if spec else []:
            base = Path(location)
            candidates = [base / "lib"] if package == "torch" else list(base.glob("*/bin"))
            for directory in candidates:
                if directory.is_dir():
                    os.add_dll_directory(str(directory))
                    os.environ["PATH"] = f"{directory}{os.pathsep}{os.environ.get('PATH', '')}"
    _dll_dirs_added = True


class FasterWhisperBackend:
    def __init__(self, model_name: str, device: str = "auto", compute_type: str = "auto",
                 download_root: Path | None = None) -> None:
        self.model_name = model_name
        self._device_pref = device
        self._compute_pref = compute_type
        self._download_root = str(download_root) if download_root else None
        self._model = None
        self._device: str | None = None
        self._compute_type: str | None = None

    # ----- availability -----
    def package_available(self) -> bool:
        return importlib.util.find_spec("faster_whisper") is not None

    def _require_package(self) -> None:
        if not self.package_available():
            raise AppError(ErrorCode.ASR_NOT_AVAILABLE, status_code=503)

    def model_installed(self) -> bool:
        if not self.package_available():
            return False
        from faster_whisper.utils import download_model

        try:
            download_model(self.model_name, local_files_only=True, cache_dir=self._download_root)
            return True
        except Exception:  # huggingface_hub raises various errors when files are missing
            return False

    def download_model(self) -> None:
        self._require_package()
        from faster_whisper.utils import download_model

        download_model(self.model_name, cache_dir=self._download_root)

    # ----- lifecycle -----
    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str | None:
        return self._device

    @property
    def compute_type(self) -> str | None:
        return self._compute_type

    def _resolve_device(self) -> tuple[str, str]:
        import ctranslate2

        cuda = ctranslate2.get_cuda_device_count() > 0
        device = "cuda" if self._device_pref in ("auto", "cuda") and cuda else "cpu"
        if self._compute_pref != "auto":
            return device, self._compute_pref
        return device, "float16" if device == "cuda" else "int8"

    def load(self) -> None:
        if self._model is not None:
            return
        self._require_package()
        if not self.model_installed():
            raise AppError(ErrorCode.ASR_MODEL_NOT_INSTALLED, status_code=409,
                           details={"modelo": self.model_name})
        _add_cuda_dll_directories()
        from faster_whisper import WhisperModel

        device, compute_type = self._resolve_device()
        started = time.perf_counter()
        try:
            self._model = WhisperModel(self.model_name, device=device, compute_type=compute_type,
                                       download_root=self._download_root, local_files_only=True)
        except Exception as exc:
            if device == "cuda":
                # Missing CUDA libraries are the most common failure on Windows: fall back to CPU.
                logger.warning("asr_cuda_load_failed_fallback_cpu", exc_info=True)
                device, compute_type = "cpu", "int8"
                self._model = WhisperModel(self.model_name, device=device, compute_type=compute_type,
                                           download_root=self._download_root, local_files_only=True)
            else:
                raise AppError(ErrorCode.MODEL_LOAD_ERROR, status_code=500) from exc
        self._device, self._compute_type = device, compute_type
        logger.info("asr_loaded", extra={"model": self.model_name, "device": device,
                                         "compute_type": compute_type,
                                         "seconds": round(time.perf_counter() - started, 2)})

    def unload(self) -> None:
        if self._model is None:
            return
        self._model = None
        gc.collect()
        if importlib.util.find_spec("torch") is not None:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        logger.info("asr_unloaded", extra={"model": self.model_name})

    # ----- inference -----
    def transcribe(self, audio_16k: np.ndarray, language: str | None,
                   initial_prompt: str | None = None) -> TranscriptionResult:
        self.load()
        started = time.perf_counter()
        segments_iter, info = self._model.transcribe(  # type: ignore[union-attr]
            audio_16k.astype(np.float32, copy=False),
            language=language,
            task="transcribe",
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,  # fewer hallucination loops on short references
            initial_prompt=initial_prompt,
        )
        segments, words = [], []
        for seg in segments_iter:  # generator: inference happens here
            segments.append(ASRSegment(start=round(seg.start, 3), end=round(seg.end, 3), text=seg.text,
                                       avg_logprob=seg.avg_logprob, no_speech_prob=seg.no_speech_prob,
                                       compression_ratio=seg.compression_ratio))
            words.extend(ASRWord(start=round(w.start, 3), end=round(w.end, 3), word=w.word,
                                 probability=round(w.probability, 4)) for w in (seg.words or []))
        return TranscriptionResult(
            text="".join(s.text for s in segments).strip(),
            language=info.language,
            language_probability=round(float(info.language_probability), 4),
            duration_s=round(float(info.duration), 3),
            segments=segments,
            words=words,
            model=self.model_name,
            device=self._device or "cpu",
            compute_type=self._compute_type or "",
            elapsed_s=round(time.perf_counter() - started, 3),
        )
