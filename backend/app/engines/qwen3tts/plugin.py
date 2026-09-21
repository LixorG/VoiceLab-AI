"""Qwen3-TTS descriptor (qwen-tts 0.1.x, verified against the official README and source).

Each variant type exposes a different API:
- Base          → generate_voice_clone(text, language, ref_audio, ref_text, x_vector_only_mode, …)  (no instruct)
- CustomVoice   → generate_custom_voice(text, speaker, language, instruct)
- VoiceDesign   → generate_voice_design(text, instruct, language)
Sampling kwargs: do_sample=True, top_k=50, top_p=1.0, temperature=0.9, repetition_penalty=1.05,
subtalker_* equivalents (library + generation_config.json) and max_new_tokens=8192 (generation_config.json of the
released checkpoints; the library fallback is 2048). There is no seed, speed, pitch or duration argument, and
`generate()` exposes no progress or stopping hooks (extra kwargs are not forwarded to the talker).
"""

from __future__ import annotations

import importlib.util
import logging
import random
import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any

import numpy as np

from app.core.errors import AppError, ErrorCode
from app.engines.base import (
    AudioCallback,
    CancelToken,
    ControlCapability,
    ControlSource,
    EngineCapabilities,
    EngineRequest,
    EngineResult,
    EngineVariant,
    GenericControl,
    LicenseInfo,
    ParameterSpec,
    ParameterTooltip,
    PreparedReference,
    ProgressCallback,
    ReferenceInput,
    SelectOption,
    TTSBackend,
    random_seed,
)
from app.engines.common import hf_files_cached, prepare_hf_cache, reference_clip
from app.engines.qwen3tts.streaming import CodeStreamer, decoder_of

logger = logging.getLogger("voicelab.engines.qwen3")

QWEN_SAMPLE_RATE = 24_000
TOKENS_PER_SECOND = 12.5
REQUIRED_FILES = ("config.json", "generation_config.json", "model.safetensors", "speech_tokenizer/config.json",
                  "speech_tokenizer/model.safetensors")
SAMPLING_PARAMS = ("do_sample", "top_k", "top_p", "temperature", "repetition_penalty", "subtalker_dosample",
                   "subtalker_top_k", "subtalker_top_p", "subtalker_temperature", "max_new_tokens")
PROMPT_CACHE_SIZE = 16

# Values accepted by the `language` argument (README), with Spanish labels.
LANGUAGE_OPTIONS = [
    ("Auto", "Detectar automáticamente"), ("Spanish", "Español"), ("English", "Inglés"), ("Chinese", "Chino"),
    ("Japanese", "Japonés"), ("Korean", "Coreano"), ("German", "Alemán"), ("French", "Francés"),
    ("Russian", "Ruso"), ("Portuguese", "Portugués"), ("Italian", "Italiano"),
]

# CustomVoice speakers documented in the README (runtime list comes from get_supported_speakers()).
SPEAKERS = [
    ("Vivian", "Mujer joven, voz brillante (nativa en chino)"),
    ("Serena", "Mujer joven, cálida y suave (nativa en chino)"),
    ("Uncle_Fu", "Hombre maduro, grave y suave (nativo en chino)"),
    ("Dylan", "Hombre joven, claro y natural (dialecto de Pekín)"),
    ("Eric", "Hombre, vivo y algo ronco (dialecto de Sichuan)"),
    ("Ryan", "Hombre dinámico y rítmico (nativo en inglés)"),
    ("Aiden", "Hombre, claro y soleado, acento estadounidense"),
    ("Ono_Anna", "Mujer juguetona y ágil (nativa en japonés)"),
    ("Sohee", "Mujer cálida y expresiva (nativa en coreano)"),
]

_SAMPLING_NOTE = "Qwen3-TTS genera token a token con muestreo aleatorio."
_PROGRESS_NOTE = ("El modelo no informa del progreso: VoiceLab lo estima contando los pasos de audio generados y "
                  "puede detener la generación a mitad si se cancela.")
_SPEED_NOTE = ("Con GPU, VoiceLab reproduce el decodificador y el predictor de códigos desde CUDA graphs. Medido en "
               "esta máquina: de unas 7 veces la duración del audio a unas 0,6 (más rápido que tiempo real), con el "
               "mismo WER, la misma similitud de voz y la misma duración. Equivalente dentro de la precisión del "
               "modelo, no idéntico bit a bit; se desactiva con QWEN_CUDA_GRAPHS=false.")
_SENTENCES_NOTE = ("Al clonar, los textos con varias frases se generan frase a frase con pausas naturales (más "
                   "largas entre párrafos). Medido con una voz real: en llamadas largas el modelo habla un 15 % más "
                   "deprisa, casi sin pausas y con la entonación más plana; frase a frase el ritmo y la entonación "
                   "quedan más cerca de la referencia, con el mismo WER y la misma similitud de voz.")
MAX_CHARS_PER_CALL = 300  # long single calls are very slow (cost grows with length) and can loop without ending
CHARS_PER_SECOND = 14.0  # typical speech rate, only used to estimate progress and a runaway limit
RUNAWAY_FACTOR = 3.0


class _Stop(Exception):  # noqa: N818 - internal control flow
    """Raised from a forward hook to leave `generate()` early."""


def _tip(what: str, effect: str, cost: str, typical: str) -> ParameterTooltip:
    return ParameterTooltip(what=what, effect=effect, cost=cost, typical=typical)


class Qwen3TTSBackend(TTSBackend):
    id = "qwen3tts"
    display_name = "Qwen3-TTS"
    description = ("Modelo de lenguaje de voz de Alibaba (Qwen). Clonación desde 3 s de audio, voces "
                   "predefinidas con instrucciones de estilo y diseño de voz por descripción.")
    license = LicenseInfo(code="Apache-2.0", weights="Apache-2.0", commercial_use="permitido",
                          url="https://github.com/QwenLM/Qwen3-TTS")
    required_packages = ("qwen-tts",)
    implementation_phase: int | None = None

    def __init__(self) -> None:
        self._model: Any = None
        self._variant: str | None = None
        self._runtime: dict[str, Any] = {}
        self._prompts: OrderedDict[tuple[str, str, str], Any] = OrderedDict()
        self._prompt_lock = threading.Lock()

    def variants(self) -> list[EngineVariant]:
        return [
            EngineVariant(id="base-1.7b", label="Clonación · 1.7B", mode="clone",
                          description="Clona una voz a partir de audio de referencia. Mayor calidad.",
                          repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base", vram_estimate_mb=5000, download_size_mb=4330),
            EngineVariant(id="base-0.6b", label="Clonación · 0.6B", mode="clone",
                          description="Clonación con menor consumo de memoria y más rápida.",
                          repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base", vram_estimate_mb=3000, download_size_mb=2400),
            EngineVariant(id="custom-voice-1.7b", label="Voces predefinidas · 1.7B", mode="custom_voice",
                          description="9 voces incluidas con control de estilo por instrucción. No clona.",
                          repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", vram_estimate_mb=5000,
                          download_size_mb=4310),
            EngineVariant(id="custom-voice-0.6b", label="Voces predefinidas · 0.6B", mode="custom_voice",
                          description="Voces incluidas, versión ligera (control de emoción más limitado).",
                          repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", vram_estimate_mb=3000,
                          download_size_mb=2390),
            EngineVariant(id="voice-design-1.7b", label="Diseño de voz · 1.7B", mode="voice_design",
                          description="Crea una voz nueva a partir de una descripción en texto. No clona.",
                          repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", vram_estimate_mb=5000,
                          download_size_mb=4310),
        ]

    # ------------------------------------------------------------------
    def capabilities(self, variant: str | None = None) -> EngineCapabilities:
        v = self.variant(variant)
        common = {
            GenericControl.LANGUAGE: ControlCapability(source=ControlSource.NATIVE, parameter="language"),
            GenericControl.SEED: ControlCapability(
                source=ControlSource.NATIVE, parameter="seed",
                reason="Se fija la semilla de PyTorch antes de generar; la reproducibilidad exacta no está "
                       "garantizada entre equipos o versiones."),
            GenericControl.TARGET_DURATION: ControlCapability(
                source=ControlSource.UNAVAILABLE,
                reason="Qwen3-TTS no permite fijar la duración; solo limita la longitud máxima generada."),
        }
        if v.mode == "clone":
            controls = {
                **common,
                GenericControl.SPEED: ControlCapability(
                    source=ControlSource.DSP,
                    reason="La clonación de Qwen3-TTS no tiene control de velocidad. Se puede ajustar en "
                           "«Posprocesado» (opcional; puede restar naturalidad)."),
                GenericControl.PITCH: ControlCapability(
                    source=ControlSource.DSP,
                    reason="El modelo no controla el tono. Se puede ajustar en «Posprocesado» (opcional; "
                           "puede alterar el timbre)."),
                GenericControl.EMOTION: ControlCapability(
                    source=ControlSource.SEGMENTATION,
                    reason="La clonación de Qwen3-TTS no acepta instrucciones de estilo. La emoción depende de "
                           "la referencia y del propio texto; se puede aproximar con referencias por fragmento."),
                GenericControl.INSTRUCTION: ControlCapability(
                    source=ControlSource.UNAVAILABLE,
                    reason="generate_voice_clone no acepta instrucciones. Usa las variantes de voces "
                           "predefinidas o de diseño de voz para controlar el estilo."),
                GenericControl.NATURALNESS: ControlCapability(
                    source=ControlSource.UNAVAILABLE,
                    reason="No existe un parámetro de naturalidad. La temperatura cambia la variabilidad, no "
                           "equivale a naturalidad."),
                GenericControl.VOICE_COLOR: ControlCapability(
                    source=ControlSource.UNAVAILABLE,
                    reason="El timbre se copia de la referencia; el modelo no permite modificarlo."),
            }
            return EngineCapabilities(
                variant=v.id, mode="clone", sample_rate=QWEN_SAMPLE_RATE,
                languages=[code for code, _ in LANGUAGE_OPTIONS if code != "Auto"],
                requires_reference_audio=True, requires_reference_text=True,
                reference_text_optional_reason="El modo «solo embedding de hablante» no necesita transcripción, "
                                               "con menor calidad según la documentación oficial.",
                reference_duration_s=(3.0, 20.0),
                reference_strategies=["best_reference", "profile", "per_segment"],
                controls=controls, supports_streaming=True, supports_batch=True, reports_progress=True,
                max_chars_per_call=MAX_CHARS_PER_CALL, sentence_chunks=True,
                reference_text_not_required_when={"clone_mode": "x_vector"},
                notes=["El prompt de clonación se calcula una vez por referencia y se reutiliza.",
                       _SENTENCES_NOTE, _SAMPLING_NOTE, _PROGRESS_NOTE, _SPEED_NOTE],
            )

        by_instruction = "Se pide dentro de la instrucción de estilo; el resultado no es un valor exacto."
        controls = {
            **common,
            GenericControl.INSTRUCTION: ControlCapability(source=ControlSource.NATIVE, parameter="instruct"),
            GenericControl.EMOTION: ControlCapability(source=ControlSource.INSTRUCTION, reason=by_instruction),
            GenericControl.SPEED: ControlCapability(source=ControlSource.INSTRUCTION, reason=by_instruction),
            GenericControl.PITCH: ControlCapability(source=ControlSource.INSTRUCTION, reason=by_instruction),
            GenericControl.NATURALNESS: ControlCapability(source=ControlSource.INSTRUCTION, reason=by_instruction),
            GenericControl.VOICE_COLOR: ControlCapability(
                source=ControlSource.INSTRUCTION if v.mode == "voice_design" else ControlSource.UNAVAILABLE,
                reason=by_instruction if v.mode == "voice_design"
                else "Las voces predefinidas tienen un timbre fijo."),
        }
        return EngineCapabilities(
            variant=v.id, mode=v.mode, sample_rate=QWEN_SAMPLE_RATE,
            languages=[code for code, _ in LANGUAGE_OPTIONS if code != "Auto"],
            requires_reference_audio=False, requires_reference_text=False,
            reference_strategies=[], controls=controls, supports_streaming=True, supports_batch=True,
            reports_progress=True, max_chars_per_call=MAX_CHARS_PER_CALL,
            notes=["Esta variante no clona voces: " + ("usa una voz incluida." if v.mode == "custom_voice"
                                                        else "crea la voz a partir de la descripción."),
                   _SAMPLING_NOTE, _PROGRESS_NOTE, _SPEED_NOTE],
        )

    # ------------------------------------------------------------------
    def get_parameters_schema(self, variant: str | None = None) -> list[ParameterSpec]:
        v = self.variant(variant)
        specs: list[ParameterSpec] = [
            ParameterSpec(
                id="language", maps_to="language", type="select", value_type="str", label="Idioma del texto",
                default="Auto", group="basic", control=GenericControl.LANGUAGE,
                options=[SelectOption(value=code, label=label) for code, label in LANGUAGE_OPTIONS],
                tooltip=_tip("Idioma en el que está escrito el texto a generar.",
                             "Indicarlo evita errores de detección, sobre todo en textos cortos o mezclados.",
                             "Sin coste.", "El idioma del texto; «Auto» si no estás seguro")),
        ]
        if v.mode == "clone":
            specs.append(ParameterSpec(
                id="clone_mode", maps_to="x_vector_only_mode", type="select", value_type="str",
                label="Modo de clonación", default="icl", group="basic",
                options=[
                    SelectOption(value="icl", label="Audio + transcripción (ICL)",
                                 description="Usa la referencia y su texto. Mayor similitud."),
                    SelectOption(value="x_vector", label="Solo embedding de hablante",
                                 description="No necesita transcripción; menor calidad según la documentación."),
                ],
                tooltip=_tip("Cómo se usa la referencia para clonar la voz.",
                             "ICL aprovecha audio y texto; el embedding solo captura rasgos generales del hablante.",
                             "ICL necesita una transcripción exacta de la referencia.",
                             "Audio + transcripción (ICL)")))
        if v.mode == "custom_voice":
            specs.append(ParameterSpec(
                id="speaker", maps_to="speaker", type="select", value_type="str", label="Voz",
                default="Vivian", group="basic", dynamic_options=True,
                options=[SelectOption(value=name, label=name.replace("_", " "), description=desc)
                         for name, desc in SPEAKERS],
                tooltip=_tip("Voz predefinida incluida en el modelo.",
                             "Cada voz es nativa de un idioma (ver descripción), aunque todas pueden hablar "
                             "los 10 idiomas.",
                             "Sin coste.", "Una voz nativa del idioma del texto")))
        if v.mode in ("custom_voice", "voice_design"):
            design = v.mode == "voice_design"
            specs.append(ParameterSpec(
                id="instruct", maps_to="instruct", type="text", value_type="str",
                label="Descripción de la voz" if design else "Instrucción de estilo",
                default=None, nullable=not design, max_length=2048, group="basic",
                control=GenericControl.INSTRUCTION,
                tooltip=_tip(
                    "Describe la voz a crear (edad, género, timbre, emoción, ritmo)." if design
                    else "Indica en lenguaje natural cómo debe hablar la voz (emoción, tono, ritmo).",
                    "El modelo interpreta la descripción; no garantiza cada detalle. Descripciones de una sola "
                    "característica suelen ser demasiado genéricas." if design
                    else "Cambia la expresión sin cambiar la voz. Según la documentación, las variantes 1.7B "
                    "tienen un control de emoción más fuerte que las 0.6B.",
                    "Sin coste apreciable.",
                    "p. ej. «Speak warmly and confidently, slightly excited but not exaggerated.»")))

        specs += [
            ParameterSpec(
                id="seed", type="seed", value_type="int", label="Semilla", default=None,
                control=GenericControl.SEED,
                tooltip=_tip("Semilla aplicada a PyTorch antes de generar (no es un argumento del modelo).",
                             "Ayuda a repetir resultados, pero la reproducibilidad no está garantizada.",
                             "Sin coste.", "Aleatoria para explorar variaciones")),
            ParameterSpec(
                id="temperature", maps_to="temperature", type="slider", value_type="float", label="Temperatura",
                default=0.9, min=0.1, max=1.5, step=0.05,
                tooltip=_tip("Aleatoriedad al elegir cada token de audio.",
                             "Bajas: más estable pero monótona; altas: más variada y con más riesgo de errores.",
                             "Sin coste.", "0.7–1.0 (0.9 por defecto)")),
            ParameterSpec(
                id="top_p", maps_to="top_p", type="slider", value_type="float", label="Top P",
                default=1.0, min=0.05, max=1.0, step=0.05,
                tooltip=_tip("Limita la elección a los tokens que suman esa probabilidad acumulada.",
                             "Valores menores reducen variaciones raras.", "Sin coste.", "1.0 (por defecto)")),
            ParameterSpec(
                id="top_k", maps_to="top_k", type="slider", value_type="int", label="Top K",
                default=50, min=1, max=200, step=1,
                tooltip=_tip("Número máximo de tokens candidatos en cada paso.",
                             "Valores bajos hacen la voz más predecible.", "Sin coste.", "50 (por defecto)")),
            ParameterSpec(
                id="repetition_penalty", maps_to="repetition_penalty", type="slider", value_type="float",
                label="Penalización por repetición", default=1.05, min=1.0, max=1.5, step=0.01,
                tooltip=_tip("Penaliza repetir tokens recientes.",
                             "Ayuda a evitar bucles; valores altos pueden alterar la pronunciación.",
                             "Sin coste.", "1.0–1.1 (1.05 por defecto)")),
            ParameterSpec(
                id="do_sample", maps_to="do_sample", type="toggle", value_type="bool", label="Muestreo aleatorio",
                default=True,
                tooltip=_tip("Si está desactivado se elige siempre el token más probable (greedy).",
                             "Sin muestreo el resultado es más determinista, pero puede sonar plano o repetirse.",
                             "Sin coste.", "Activado")),
            ParameterSpec(
                id="subtalker_temperature", maps_to="subtalker_temperature", type="slider", value_type="float",
                label="Temperatura del detalle acústico", default=0.9, min=0.1, max=1.5, step=0.05,
                tooltip=_tip("Aleatoriedad de los códigos residuales que aportan el detalle fino del sonido.",
                             "Afecta a textura y matices más que al contenido.", "Sin coste.",
                             "0.9 (por defecto)")),
            ParameterSpec(
                id="subtalker_top_p", maps_to="subtalker_top_p", type="slider", value_type="float",
                label="Top P del detalle acústico", default=1.0, min=0.05, max=1.0, step=0.05,
                tooltip=_tip("Top P aplicado a los códigos residuales.", "Valores menores reducen variaciones.",
                             "Sin coste.", "1.0 (por defecto)")),
            ParameterSpec(
                id="subtalker_top_k", maps_to="subtalker_top_k", type="slider", value_type="int",
                label="Top K del detalle acústico", default=50, min=1, max=200, step=1,
                tooltip=_tip("Top K aplicado a los códigos residuales.", "Valores bajos, más predecible.",
                             "Sin coste.", "50 (por defecto)")),
            ParameterSpec(
                id="subtalker_dosample", maps_to="subtalker_dosample", type="toggle", value_type="bool",
                label="Muestreo del detalle acústico", default=True,
                tooltip=_tip("Muestreo aleatorio para los códigos residuales.",
                             "Desactivado puede sonar más artificial.", "Sin coste.", "Activado")),
            ParameterSpec(
                id="max_new_tokens", maps_to="max_new_tokens", type="slider", value_type="int",
                label="Longitud máxima", default=8192, min=256, max=8192, step=64, unit="tokens",
                tooltip=_tip("Límite de tokens de audio generados (12,5 tokens por segundo de audio).",
                             "Evita bucles sin fin; si es demasiado bajo corta el audio.",
                             "Más memoria y tiempo solo si el audio realmente es largo.",
                             "8192 (≈ 11 min, valor de generation_config.json del modelo)")),
        ]
        return specs

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------
    def weights_installed(self, variant: str) -> bool:
        if not self.is_installed():
            return False
        return hf_files_cached(self.variant(variant).repo_id or "", REQUIRED_FILES)

    def download_weights(self, variant: str) -> None:
        from huggingface_hub import snapshot_download

        prepare_hf_cache()
        snapshot_download(repo_id=self.variant(variant).repo_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def loaded_variant(self) -> str | None:
        return self._variant

    def load(self, variant: str, device: str, dtype: str) -> None:
        v = self.variant(variant)
        if self._model is not None and self._variant == v.id:
            return
        import torch
        from huggingface_hub import snapshot_download
        from qwen_tts import Qwen3TTSModel

        from app.core.config import get_settings

        path = snapshot_download(repo_id=v.repo_id, local_files_only=True)
        if device == "cuda":
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            device_map = "cuda:0"
        else:
            torch_dtype, device_map = torch.float32, device
        attn = get_settings().attn_implementation
        if attn == "auto":
            attn = "flash_attention_2" if device == "cuda" and importlib.util.find_spec("flash_attn") else "sdpa"

        self.unload()
        self._model = Qwen3TTSModel.from_pretrained(path, device_map=device_map, dtype=torch_dtype,
                                                    attn_implementation=attn)
        model_type = getattr(self._model.model, "tts_model_type", None)
        expected = {"clone": "base", "custom_voice": "custom_voice", "voice_design": "voice_design"}[v.mode]
        if model_type != expected:
            self.unload()
            raise AppError(ErrorCode.MODEL_LOAD_ERROR, status_code=500,
                           message="Los pesos descargados no corresponden a la variante seleccionada.",
                           details={"esperado": expected, "encontrado": model_type})
        self._variant = v.id
        self._runtime = {"dtype": str(torch_dtype).replace("torch.", ""), "attention": attn,
                         "acceleration": self._accelerate(device)}

    def _accelerate(self, device: str) -> str:
        """Speed up each audio frame (see fast_predictor): CUDA graphs on GPU, else a direct predictor loop.

        `cuda_graphs` = talker decoder and code predictor graphed; `cuda_graphs_partial` = only one of them could
        be captured; `direct_loop` = no graphs (bit-identical to the library); `none` = unexpected model layout.
        """
        from app.core.config import get_settings
        from app.engines.qwen3tts import fast_predictor

        if device == "cuda" and get_settings().qwen_cuda_graphs:
            talker = fast_predictor.install_talker_graphs(self._model)
            predictor = fast_predictor.install_graphs(self._model)
            if talker or predictor:
                self._check_random_generator()
                if not predictor:
                    fast_predictor.install(self._model)
                return "cuda_graphs" if talker and predictor else "cuda_graphs_partial"
        return "direct_loop" if fast_predictor.install(self._model) else "none"

    def _check_random_generator(self) -> None:
        """A capture that fails half-way can leave CUDA's random generator unusable: refuse to go on silently."""
        import torch

        try:
            torch.multinomial(torch.ones(1, 4, device="cuda"), 1)
        except RuntimeError as exc:
            self.unload()
            raise AppError(ErrorCode.MODEL_LOAD_ERROR, status_code=500,
                           message="La aceleración de Qwen3-TTS dejó la GPU en un estado no válido. Pon "
                                   "QWEN_CUDA_GRAPHS=false en el archivo .env y reinicia VoiceLab.") from exc

    def unload(self) -> None:
        self._model = None
        self._variant = None
        self._runtime = {}
        with self._prompt_lock:
            self._prompts.clear()

    def supported_speakers(self) -> list[str] | None:
        if self._model is None:
            return None
        speakers = self._model.get_supported_speakers()
        return sorted(speakers) if speakers else None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def prepare_reference(self, reference: ReferenceInput, variant: str) -> PreparedReference:
        clip, key = reference_clip(reference)
        return PreparedReference(engine=self.id, variant=variant, cache_key=key,
                                 payload={"audio_path": str(clip), "text": (reference.text or "").strip()})

    def _voice_clone_prompt(self, reference: PreparedReference, clone_mode: str) -> Any:
        """create_voice_clone_prompt once per (variant, clip, mode); reused across generations (LRU in memory)."""
        import soundfile as sf

        x_vector_only = clone_mode == "x_vector"
        text = reference.payload.get("text") or None
        if not x_vector_only and not text:
            raise AppError(ErrorCode.REFERENCE_TEXT_REQUIRED, status_code=422)
        cache_key = (self._variant or "", reference.cache_key, f"{clone_mode}|{'' if x_vector_only else text}")
        with self._prompt_lock:
            if cache_key in self._prompts:
                self._prompts.move_to_end(cache_key)
                return self._prompts[cache_key]
        audio, sr = sf.read(reference.payload["audio_path"], dtype="float32", always_2d=False)
        items = self._model.create_voice_clone_prompt(ref_audio=(audio, sr), ref_text=None if x_vector_only else text,
                                                      x_vector_only_mode=x_vector_only)
        with self._prompt_lock:
            self._prompts[cache_key] = items
            while len(self._prompts) > PROMPT_CACHE_SIZE:
                self._prompts.popitem(last=False)
        return items

    def generate(self, request: EngineRequest, progress: ProgressCallback, cancel: CancelToken,
                 on_audio: AudioCallback | None = None) -> EngineResult:
        if self._model is None or self._variant != request.variant:
            raise AppError(ErrorCode.MODEL_LOAD_ERROR, status_code=500, message="El modelo no está cargado.")
        v = self.variant(request.variant)
        params = self.validate_parameters(request.params, v.id)
        seed = params["seed"] if params["seed"] is not None else random_seed()
        self._seed_everything(seed)
        sampling = {k: params[k] for k in SAMPLING_PARAMS}

        if cancel.cancelled:
            raise AppError(ErrorCode.JOB_CANCELLED, status_code=409)
        # Expected audio tokens from the text length: drives the estimated progress and a runaway limit, so a
        # model that never emits the end token cannot keep the GPU busy for up to max_new_tokens (~11 min of audio).
        expected = max(25, int(len(request.text) / CHARS_PER_SECOND * TOKENS_PER_SECOND))
        limit = min(params["max_new_tokens"], int(expected * RUNAWAY_FACTOR) + 60)
        sampling["max_new_tokens"] = limit
        warnings: list[str] = []

        def run(fn, prefix=None, **kwargs):  # noqa: ANN001, ANN202
            with self._step_hook(expected, progress, cancel), self._stream_hook(on_audio, prefix):
                return fn(**kwargs, **sampling)

        if v.mode == "clone":
            if request.reference is None:
                raise AppError(ErrorCode.REFERENCE_REQUIRED, status_code=422)
            progress(0.05, "Codificando la referencia…")
            prompt = self._voice_clone_prompt(request.reference, params["clone_mode"])
            if cancel.cancelled:
                raise AppError(ErrorCode.JOB_CANCELLED, status_code=409)
            item = prompt[0] if prompt else None
            prefix = getattr(item, "ref_code", None) if getattr(item, "icl_mode", False) else None
            wavs, sr = run(self._model.generate_voice_clone, prefix=prefix, text=request.text,
                           language=params["language"], voice_clone_prompt=prompt)
        elif v.mode == "custom_voice":
            wavs, sr = run(self._model.generate_custom_voice, text=request.text, speaker=params["speaker"],
                           language=params["language"], instruct=params["instruct"] or None)
        else:
            wavs, sr = run(self._model.generate_voice_design, text=request.text, instruct=params["instruct"],
                           language=params["language"])

        audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        if audio.size / sr * TOKENS_PER_SECOND >= limit * 0.98:
            if limit < params["max_new_tokens"]:
                warnings.append("El modelo no terminaba la frase y se detuvo por seguridad (posible bucle): revisa "
                                "el final del audio o prueba otra semilla.")
            else:
                warnings.append("Se alcanzó la longitud máxima de tokens: el audio puede estar cortado.")
        progress(1.0, None)
        return EngineResult(audio=audio, sample_rate=int(sr), seed=seed, effective_params={**params, "seed": seed},
                            warnings=warnings, details=dict(self._runtime))

    @contextmanager
    def _step_hook(self, expected_tokens: int, progress: ProgressCallback, cancel: CancelToken):  # noqa: ANN202
        """Count talker decoding steps: estimated progress + cancellation in the middle of `generate()`."""
        talker = getattr(getattr(self._model, "model", None), "talker", None)
        if talker is None or not hasattr(talker, "register_forward_pre_hook"):
            progress(0.1, "Generando voz…")
            yield
            return
        steps = {"n": 0}

        def hook(_module, _args):  # noqa: ANN001, ANN202
            if cancel.cancelled:
                raise _Stop
            steps["n"] += 1
            if steps["n"] % 12 == 0:  # ~1 s of audio
                fraction = min(0.95, 0.1 + 0.85 * steps["n"] / expected_tokens)
                progress(fraction, f"Generando voz · ~{steps['n'] / TOKENS_PER_SECOND:.0f} s de audio (estimado)…")

        handle = talker.register_forward_pre_hook(hook)
        progress(0.1, "Generando voz…")
        try:
            yield
        except _Stop as exc:
            raise AppError(ErrorCode.JOB_CANCELLED, status_code=409) from exc
        finally:
            handle.remove()

    @contextmanager
    def _stream_hook(self, on_audio: AudioCallback | None, prefix: Any):  # noqa: ANN202
        """Live audio: decode the codec frames of each talker step in small chunks (see streaming.py)."""
        talker = getattr(getattr(self._model, "model", None), "talker", None)
        decoder = decoder_of(self._model) if on_audio is not None else None
        if on_audio is None or decoder is None or talker is None or not hasattr(talker, "register_forward_hook"):
            yield
            return
        decode, samples_per_frame, rate = decoder
        streamer = CodeStreamer(decode, on_audio, prefix=prefix, samples_per_frame=samples_per_frame,
                                sample_rate=rate)

        def hook(_module, _args, output):  # noqa: ANN001, ANN202
            hidden = getattr(output, "hidden_states", None)
            codes = hidden[1] if isinstance(hidden, tuple) and len(hidden) > 1 else None
            if codes is not None and codes.shape[0] == 1:
                streamer.add(codes[0])

        handle = talker.register_forward_hook(hook)
        try:
            yield
            streamer.flush()
        finally:
            handle.remove()

    @staticmethod
    def _seed_everything(seed: int) -> None:
        import torch

        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


ENGINE_CLASS = Qwen3TTSBackend
