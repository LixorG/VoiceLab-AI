"""Shared description for engines built on `f5_tts.api.F5TTS` (F5-TTS and the E2-TTS reproduction).

Parameters mirror `F5TTS.infer()` (f5-tts 1.1.x, verified in FASE 0): nfe_step, cfg_strength,
sway_sampling_coef, speed, fix_duration, remove_silence, cross_fade_duration, target_rms, seed.
UI ranges follow the official Gradio app or are conservative; defaults are the library defaults.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from app.core.errors import AppError, ErrorCode
from app.engines.base import (
    CancelToken,
    ControlCapability,
    ControlSource,
    EngineCapabilities,
    EngineRequest,
    EngineResult,
    EngineVariant,
    GenericControl,
    ParameterSpec,
    ParameterTooltip,
    PreparedReference,
    ProgressCallback,
    ReferenceInput,
    TTSBackend,
    random_seed,
)
from app.engines.common import hf_files_cached, prepare_hf_cache, reference_clip
from app.engines.custom import CustomVariant, get_custom_variants, is_custom

logger = logging.getLogger("voicelab.engines.f5")

CUSTOM_DESCRIPTION = "Checkpoint personalizado añadido por ti."

F5_SAMPLE_RATE = 24_000
VOCOS_REPO = "charactr/vocos-mel-24khz"
VOCOS_FILES = ("config.yaml", "pytorch_model.bin")


class _ProgressAdapter:
    """F5-TTS calls `progress.tqdm(iterable)` over text batches; forward the fraction to our callback."""

    def __init__(self, callback: ProgressCallback, cancel: CancelToken) -> None:
        self._callback = callback
        self._cancel = cancel

    def tqdm(self, iterable: Any) -> Any:
        items = list(iterable)
        total = max(1, len(items))
        for index, item in enumerate(items, start=1):
            if self._cancel.cancelled:
                raise AppError(ErrorCode.JOB_CANCELLED, status_code=409)
            yield item
            self._callback(index / total, f"Generando fragmento {index} de {total}…" if total > 1 else None)


class FlowMatchingBackend(TTSBackend):
    required_packages = ("f5-tts",)
    implementation_phase: int | None = None
    supports_custom_checkpoints = True
    #: variant id -> (Hugging Face repo, checkpoint file). Mirrors f5_tts.api.F5TTS naming.
    checkpoints: dict[str, tuple[str, str]] = {}

    def __init__(self) -> None:
        self._model: Any = None
        self._variant: str | None = None

    def builtin_variants(self) -> list[EngineVariant]:
        """Official checkpoints of this engine; subclasses implement it."""
        raise NotImplementedError

    def variants(self) -> list[EngineVariant]:
        custom = [EngineVariant(id=v.id, label=v.name, description=v.notes or CUSTOM_DESCRIPTION,
                                source="custom", base_variant=v.base_variant, languages=v.languages or None,
                                repo_id=v.repo_id, vram_estimate_mb=self._base_vram(v.base_variant))
                  for v in get_custom_variants().for_engine(self.id)]
        return [*self.builtin_variants(), *custom]

    def _base_vram(self, base_variant: str) -> int | None:
        return next((v.vram_estimate_mb for v in self.builtin_variants() if v.id == base_variant), None)

    def _custom(self, variant: str | None) -> CustomVariant | None:
        v = self.variant(variant)
        return get_custom_variants().get(self.id, v.id) if is_custom(v.id) else None

    def capabilities(self, variant: str | None = None) -> EngineCapabilities:
        v = self.variant(variant)
        return EngineCapabilities(
            variant=v.id,
            mode="clone",
            sample_rate=F5_SAMPLE_RATE,
            languages=v.languages or ["en", "zh"],
            requires_reference_audio=True,
            requires_reference_text=True,
            reference_duration_s=(3.0, 12.0),
            reference_strategies=["best_reference", "profile", "per_segment"],
            deterministic_seed=True,
            controls={
                GenericControl.SPEED: ControlCapability(source=ControlSource.NATIVE, parameter="speed"),
                GenericControl.TARGET_DURATION: ControlCapability(source=ControlSource.NATIVE,
                                                                  parameter="fix_duration"),
                GenericControl.SEED: ControlCapability(source=ControlSource.NATIVE, parameter="seed"),
                GenericControl.EMOTION: ControlCapability(
                    source=ControlSource.SEGMENTATION,
                    reason=f"{self.display_name} no tiene control de emoción. Se aproxima generando cada "
                           "fragmento con una referencia grabada con esa emoción."),
                GenericControl.PITCH: ControlCapability(
                    source=ControlSource.DSP,
                    reason="El modelo no controla el tono. Se puede ajustar en «Posprocesado» (opcional; "
                           "puede alterar el timbre)."),
                GenericControl.NATURALNESS: ControlCapability(
                    source=ControlSource.UNAVAILABLE,
                    reason="No existe un parámetro de naturalidad en este modelo. Depende sobre todo de la "
                           "calidad de la referencia y de su transcripción."),
                GenericControl.VOICE_COLOR: ControlCapability(
                    source=ControlSource.UNAVAILABLE,
                    reason="El timbre se copia de la referencia; el modelo no permite modificarlo."),
                GenericControl.INSTRUCTION: ControlCapability(
                    source=ControlSource.UNAVAILABLE,
                    reason="Este modelo no acepta instrucciones de estilo en lenguaje natural."),
                GenericControl.LANGUAGE: ControlCapability(
                    source=ControlSource.UNAVAILABLE,
                    reason="No hay selector de idioma: lo determinan el texto y el checkpoint. Los pesos "
                           "oficiales están entrenados en inglés y chino."),
            },
            notes=[
                "La transcripción de la referencia debe coincidir exactamente con el audio.",
                "Referencias de más de ~12 s se recortan automáticamente en el modelo.",
                "Para otros idiomas (p. ej. español) se necesita un checkpoint ajustado por la comunidad.",
                *(["Checkpoint personalizado: los idiomas, la calidad y la licencia son los que declaraste al "
                   "añadirlo; no hay forma de comprobarlos automáticamente."] if v.source == "custom" else []),
            ],
        )

    def get_parameters_schema(self, variant: str | None = None) -> list[ParameterSpec]:
        self.variant(variant)
        return [
            ParameterSpec(
                id="speed", maps_to="speed", type="slider", value_type="float", label="Velocidad de habla",
                default=1.0, min=0.3, max=2.0, step=0.05, unit="x", group="basic", control=GenericControl.SPEED,
                tooltip=ParameterTooltip(
                    what="Ajusta la duración que el modelo asigna al texto (control nativo, no procesamiento "
                         "posterior).",
                    effect="Valores bajos alargan el habla y altos la aceleran. Los extremos suelen perder "
                           "naturalidad.",
                    cost="Sin coste adicional.",
                    typical="0.9–1.1 (1.0 por defecto)")),
            ParameterSpec(
                id="seed", maps_to="seed", type="seed", value_type="int", label="Semilla",
                default=None, group="advanced", control=GenericControl.SEED,
                tooltip=ParameterTooltip(
                    what="Número que fija el ruido inicial de la generación.",
                    effect="La misma semilla con los mismos parámetros y referencia reproduce el mismo resultado.",
                    cost="Sin coste.",
                    typical="Aleatoria para explorar; fija para repetir una generación")),
            ParameterSpec(
                id="nfe_steps", maps_to="nfe_step", type="slider", value_type="int", label="Pasos de inferencia",
                default=32, min=4, max=64, step=2, presets={"fast": 16, "balanced": 32, "high_fidelity": 48},
                tooltip=ParameterTooltip(
                    what="Controla el número de pasos utilizados durante la generación.",
                    effect="Más pasos pueden aumentar el tiempo de procesamiento y no garantizan necesariamente "
                           "una mejora proporcional.",
                    cost="El tiempo de generación crece de forma casi lineal con los pasos.",
                    typical="16–48 (32 por defecto)")),
            ParameterSpec(
                id="cfg_strength", maps_to="cfg_strength", type="slider", value_type="float",
                label="Intensidad de guía (CFG)", default=2.0, min=0.0, max=5.0, step=0.1,
                tooltip=ParameterTooltip(
                    what="Peso de la guía sin clasificador: cuánto se ciñe la generación al texto y a la referencia.",
                    effect="Valores altos pueden sonar forzados o con artefactos; bajos, menos inteligibles.",
                    cost="Sin coste adicional apreciable.",
                    typical="1.5–2.5 (2.0 por defecto)")),
            ParameterSpec(
                id="sway_sampling_coef", maps_to="sway_sampling_coef", type="slider", value_type="float",
                label="Coeficiente de sway sampling", default=-1.0, min=-1.0, max=1.0, step=0.1,
                tooltip=ParameterTooltip(
                    what="Reparte los pasos de inferencia a lo largo del proceso de generación.",
                    effect="Valores negativos concentran pasos al principio, donde los autores observaron más "
                           "beneficio.",
                    cost="Sin coste.",
                    typical="−1 (por defecto, recomendado por los autores)")),
            ParameterSpec(
                id="cross_fade_duration", maps_to="cross_fade_duration", type="slider", value_type="float",
                label="Fundido entre fragmentos", default=0.15, min=0.0, max=1.0, step=0.01, unit="s",
                tooltip=ParameterTooltip(
                    what="Duración del fundido cruzado entre los fragmentos en que el modelo divide textos largos.",
                    effect="Suaviza las uniones; valores altos pueden solapar sílabas.",
                    cost="Sin coste.",
                    typical="0.1–0.2 s (0.15 por defecto)")),
            ParameterSpec(
                id="remove_silence", maps_to="remove_silence", type="toggle", value_type="bool",
                label="Eliminar silencios largos", default=False,
                tooltip=ParameterTooltip(
                    what="Recorta silencios de 1 s o más del audio generado.",
                    effect="Puede alterar el ritmo y las pausas naturales.",
                    cost="Mínimo.",
                    typical="Desactivado")),
            ParameterSpec(
                id="fix_duration", maps_to="fix_duration", type="number", value_type="float",
                label="Duración total fija", default=None, nullable=True, min=1.0, max=300.0, step=0.5, unit="s",
                control=GenericControl.TARGET_DURATION,
                tooltip=ParameterTooltip(
                    what="Fuerza la duración total en segundos, incluida la parte correspondiente a la referencia.",
                    effect="Sustituye al cálculo por velocidad. Duraciones poco realistas degradan la voz.",
                    cost="Sin coste.",
                    typical="Vacío (automático)")),
            ParameterSpec(
                id="target_rms", maps_to="target_rms", type="slider", value_type="float",
                label="Nivel RMS de la referencia", default=0.1, min=0.01, max=0.3, step=0.01,
                tooltip=ParameterTooltip(
                    what="Nivel al que se normaliza la referencia antes de generar.",
                    effect="Afecta al volumen del resultado; rara vez necesita cambios.",
                    cost="Sin coste.",
                    typical="0.1 (por defecto)")),
        ]

    # ------------------------------------------------------------------
    # Weights (only local cache checks; downloads happen on explicit request)
    # ------------------------------------------------------------------
    def _files(self, variant: str) -> list[tuple[str, str]]:
        """Hugging Face files this variant needs; a checkpoint stored on disk only needs the vocoder."""
        custom = self._custom(variant)
        vocoder = [(VOCOS_REPO, f) for f in VOCOS_FILES]
        if custom is None:
            return [self.checkpoints[self.variant(variant).id], *vocoder]
        if custom.from_hub:
            vocab = [(custom.repo_id, custom.vocab_file)] if custom.vocab_file else []
            return [(custom.repo_id, custom.ckpt_file), *vocab, *vocoder]  # type: ignore[list-item]
        return vocoder

    def weights_installed(self, variant: str) -> bool:
        if not self.is_installed():
            return False
        custom = self._custom(variant)
        if custom is not None and not custom.from_hub:
            path = custom.resolved_path()
            if path is None or not path.exists():
                return False
        return all(hf_files_cached(repo, (filename,)) for repo, filename in self._files(variant))

    def download_weights(self, variant: str) -> None:
        from huggingface_hub import hf_hub_download

        custom = self._custom(variant)
        if custom is not None and not custom.from_hub:
            path = custom.resolved_path()
            if path is None or not path.exists():
                raise AppError(ErrorCode.MODEL_NOT_INSTALLED, status_code=409,
                               message="El archivo del checkpoint personalizado ya no está en su ruta.",
                               details={"ruta": custom.local_path})
        prepare_hf_cache()
        for repo, filename in self._files(variant):
            hf_hub_download(repo_id=repo, filename=filename)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def loaded_variant(self) -> str | None:
        return self._variant

    def load(self, variant: str, device: str, dtype: str) -> None:
        v = self.variant(variant).id
        if self._model is not None and self._variant == v:
            return
        from f5_tts.api import F5TTS
        from huggingface_hub import hf_hub_download

        custom = self._custom(v)
        architecture, vocab = v, ""
        if custom is None:
            repo, filename = self.checkpoints[v]
            ckpt = hf_hub_download(repo_id=repo, filename=filename, local_files_only=True)
        else:
            # The architecture name must stay one of f5-tts' own configs; the weights are the user's.
            architecture = custom.base_variant
            vocab = custom.vocab_path or ""
            if custom.from_hub:
                ckpt = hf_hub_download(repo_id=custom.repo_id, filename=custom.ckpt_file, local_files_only=True)
                if custom.vocab_file:
                    vocab = hf_hub_download(repo_id=custom.repo_id, filename=custom.vocab_file,
                                            local_files_only=True)
            else:
                ckpt = str(custom.resolved_path())
        vocos_dir = Path(hf_hub_download(repo_id=VOCOS_REPO, filename=VOCOS_FILES[0], local_files_only=True)).parent
        self.unload()
        self._model = F5TTS(model=architecture, ckpt_file=ckpt, vocab_file=vocab,
                            vocoder_local_path=str(vocos_dir), device=device)
        self._variant = v

    def unload(self) -> None:
        self._model = None
        self._variant = None

    def prepare_reference(self, reference: ReferenceInput, variant: str) -> PreparedReference:
        """Cut the selected range from the processed 24 kHz WAV (cached by content + range)."""
        if not reference.text or not reference.text.strip():
            raise AppError(ErrorCode.REFERENCE_TEXT_REQUIRED, status_code=422)
        clip, key = reference_clip(reference)
        return PreparedReference(engine=self.id, variant=variant, cache_key=key,
                                 payload={"audio_path": str(clip), "text": reference.text.strip()})

    def generate(self, request: EngineRequest, progress: ProgressCallback, cancel: CancelToken) -> EngineResult:
        if self._model is None or self._variant != request.variant:
            raise AppError(ErrorCode.MODEL_LOAD_ERROR, status_code=500, message="El modelo no está cargado.")
        if request.reference is None:
            raise AppError(ErrorCode.REFERENCE_REQUIRED, status_code=422)
        params = self.validate_parameters(request.params, request.variant)
        seed = params["seed"] if params["seed"] is not None else random_seed()
        if cancel.cancelled:
            raise AppError(ErrorCode.JOB_CANCELLED, status_code=409)

        ref = request.reference.payload
        wav, sr, _spec = self._model.infer(
            ref_file=ref["audio_path"],
            ref_text=ref["text"],
            gen_text=request.text,
            show_info=lambda *args, **_: logger.debug("f5tts: %s", " ".join(str(a) for a in args)),
            progress=_ProgressAdapter(progress, cancel),
            target_rms=params["target_rms"],
            cross_fade_duration=params["cross_fade_duration"],
            sway_sampling_coef=params["sway_sampling_coef"],
            cfg_strength=params["cfg_strength"],
            nfe_step=params["nfe_steps"],
            speed=params["speed"],
            fix_duration=params["fix_duration"],
            remove_silence=False,
            seed=seed,
        )
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        if params["remove_silence"]:
            audio = self._remove_silence(audio, sr)
        return EngineResult(audio=audio, sample_rate=int(sr), seed=seed, effective_params={**params, "seed": seed})

    @staticmethod
    def _remove_silence(audio: np.ndarray, sr: int) -> np.ndarray:
        """F5-TTS only removes silence when exporting to a file, so use its own routine on a temp file."""
        import soundfile as sf
        from f5_tts.infer.utils_infer import remove_silence_for_generated_wav

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.wav"
            sf.write(path, audio, sr)
            remove_silence_for_generated_wav(str(path))
            trimmed, _ = sf.read(path, dtype="float32", always_2d=False)
        return trimmed
