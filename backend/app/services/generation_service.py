"""Generation requests → validated Generation rows (+ segments) → background jobs → WAV files.

Text goes through SpeechMarkup → planner (engine-aware) before anything is queued, so every problem (invalid tag,
missing transcript, reference too long…) is reported synchronously in Spanish. Simple texts use a single engine
call; texts with pauses, emotions or styles are generated segment by segment and joined with crossfades.
"""

from __future__ import annotations

import inspect
import logging
import random
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from sqlmodel import Session, select

from app.audio.storage import AudioStorage
from app.core.config import Settings, get_settings
from app.core.database import get_engine
from app.core.errors import AppError, ErrorCode
from app.engines.base import SEED_MAX, ControlSource, EngineRequest, GenericControl, ReferenceInput, TTSBackend
from app.engines.manager import ModelManager, get_model_manager
from app.generation.assembly import SegmentAudio, assemble
from app.generation.markup import MarkupError, ParsedMarkup, Style, TextRun, parse
from app.generation.planner import GenerationPlan, PlanError, plan_generation
from app.generation.streaming import StreamWriter, silence
from app.models.entities import Generation, GenerationSegment, ReferenceAudio, Transcript
from app.models.enums import JobStatus, ReferenceStatus
from app.postprocess.config import PostProcessConfig
from app.schemas.generation import (
    MAX_BATCH,
    BatchCreate,
    GenerationCreate,
    GenerationPlanRead,
    GenerationPostProcess,
    GenerationRead,
    GenerationReference,
    PlannedSegmentRead,
)
from app.services import mastering
from app.workers.asyncio_queue import AsyncioJobQueue, JobContext
from app.workers.base import JobSpec, JobState

logger = logging.getLogger("voicelab.generation")

JOB_KIND = "generation"
PREVIEW_MAX_CHARS = 160
REFERENCE_DURATION_TOLERANCE_S = 0.5
INSTRUCTION_MAX_CHARS = 2048


def preview_text(text: str) -> str:
    """First sentence (or first PREVIEW_MAX_CHARS characters cut at a word boundary)."""
    match = re.search(r"^.+?[.!?。！？…](\s|$)", text, flags=re.S)
    sentence = (match.group(0) if match else text).strip()
    if len(sentence) <= PREVIEW_MAX_CHARS:
        return sentence
    cut = sentence[:PREVIEW_MAX_CHARS].rsplit(" ", 1)[0]
    return (cut or sentence[:PREVIEW_MAX_CHARS]).strip()


@dataclass
class BuiltSegment:
    index: int
    text: str
    params: dict[str, Any]
    emotion: str | None
    emotion_via: str | None
    instruction: str | None
    pause_before_ms: int
    pause_after_ms: int
    reference: dict | None


@dataclass
class BuiltGeneration:
    engine: TTSBackend
    variant: str
    params: dict[str, Any]
    text: str
    segments: list[BuiltSegment]
    segmented: bool
    reference_id: str | None
    warnings: list[str] = field(default_factory=list)
    text_changes: list[dict[str, str]] = field(default_factory=list)
    normalize_language: str | None = None


def body_of_batch(body: BatchCreate, text: str, variant: str | None,
                  params: dict[str, Any]) -> GenerationCreate:
    return GenerationCreate(**{**body.model_dump(exclude={"texts", "variants", "repeat"}),
                               "text": text, "variant": variant, "params": params})


class GenerationService:
    def __init__(self, session: Session, settings: Settings, models: ModelManager, queue: AsyncioJobQueue) -> None:
        self.session = session
        self.settings = settings
        self.models = models
        self.queue = queue
        self.output_dir = settings.data_dir / "generated"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ build (shared by plan / create)
    def build(self, body: GenerationCreate, check_install: bool = True) -> BuiltGeneration:
        engine = self.models.registry.get(body.engine)
        variant = engine.variant(body.variant).id
        if engine.implementation_phase is not None:
            raise AppError(ErrorCode.NOT_IMPLEMENTED, status_code=501,
                           message=f"La generación con {engine.display_name} estará disponible en la fase "
                                   f"{engine.implementation_phase}.")
        if check_install:
            status = self.models.status(engine.id, variant)
            if not status.package_installed:
                raise AppError(ErrorCode.MODEL_NOT_INSTALLED, status_code=409,
                               details={"paquetes": status.missing_packages})
            if not status.weights_installed:
                raise AppError(ErrorCode.MODEL_NOT_INSTALLED, status_code=409,
                               message="Los pesos del modelo no están descargados. "
                                       "Descárgalos desde el panel del motor.",
                               details={"motor": engine.id, "variante": variant})

        params = engine.validate_parameters(body.params, variant)
        caps = engine.capabilities(variant)
        needs_text = engine.requires_reference_text(params, variant)
        reference_id, scope_refs = self._reference_scope(body, caps.requires_reference_audio)

        try:
            parsed = parse(body.text) if body.markup else ParsedMarkup(events=[TextRun(body.text, Style())])
        except MarkupError as exc:
            raise AppError(ErrorCode.MARKUP_ERROR, status_code=422, message=exc.message,
                           details={"posicion": exc.position, "etiqueta": exc.tag}) from exc

        text_changes, normalize_language = self._normalize(parsed, body, params)
        tagged = self._tagged_references(scope_refs, needs_text) if caps.requires_reference_audio else {}
        try:
            plan = plan_generation(parsed, caps, engine.display_name, body.emotion, body.intensity, set(tagged))
        except PlanError as exc:
            raise AppError(ErrorCode.VALIDATION_ERROR, status_code=422, message=str(exc)) from exc
        if body.preview:
            plan = self._preview_plan(plan)

        warnings = list(plan.warnings)
        snapshots: dict[str, dict] = {}

        def snapshot(ref_id: str | None) -> dict | None:
            if not caps.requires_reference_audio:
                return None
            key = ref_id or ""
            if key not in snapshots:
                snap, ref_warnings = self._resolve_reference(ref_id, needs_text, caps.reference_duration_s)
                snapshots[key] = snap
                warnings.extend(w for w in ref_warnings if w not in warnings)
            return snapshots[key]

        instruct_param = caps.controls[GenericControl.INSTRUCTION].parameter
        segmented = not plan.is_simple
        segments: list[BuiltSegment] = []
        for seg in plan.segments:
            seg_params = dict(params)
            if seg.instruction and instruct_param:
                base = str(params.get(instruct_param) or "").strip()
                combined = f"{base.rstrip('.')}. {seg.instruction}" if base else seg.instruction
                seg_params[instruct_param] = combined[:INSTRUCTION_MAX_CHARS]
                seg_params = engine.validate_parameters(seg_params, variant)
            ref_id = tagged[seg.emotion] if seg.emotion_via == "reference" and seg.emotion in tagged else reference_id
            segments.append(BuiltSegment(
                index=seg.index, text=seg.text, params=seg_params, emotion=seg.emotion, emotion_via=seg.emotion_via,
                instruction=seg.instruction, pause_before_ms=seg.pause_before_ms, pause_after_ms=seg.pause_after_ms,
                reference=snapshot(ref_id)))

        if not segmented:
            text = segments[0].text
        else:
            text = " ".join(s.text for s in segments) if body.preview else body.text
        return BuiltGeneration(engine=engine, variant=variant, params=params, text=text, segments=segments,
                               segmented=segmented,
                               reference_id=reference_id if caps.requires_reference_audio else None,
                               warnings=warnings, text_changes=text_changes, normalize_language=normalize_language)

    def _normalize(self, parsed: ParsedMarkup, body: GenerationCreate,
                   params: dict[str, Any]) -> tuple[list[dict[str, str]], str | None]:
        """Rewrite numbers/abbreviations and apply the pronunciation dictionary to the text runs (not the tags)."""
        if not body.normalize:
            return [], None
        from app.services.pronunciation_service import PronunciationService
        from app.text.normalization import normalize_text, unique_changes

        service = PronunciationService(self.session)
        language = service.resolve_language(params, body.profile_id)
        dictionary = service.dictionary(body.profile_id)
        changes = []
        for event in parsed.events:
            if isinstance(event, TextRun):
                event.text, run_changes = normalize_text(event.text, language, dictionary)
                changes.extend(run_changes)
        return [{"original": c.original, "replacement": c.replacement, "kind": c.kind}
                for c in unique_changes(changes)], language

    @staticmethod
    def _preview_plan(plan: GenerationPlan) -> GenerationPlan:
        first = plan.segments[0]
        first.text = preview_text(first.text)
        first.pause_before_ms = first.pause_after_ms = 0
        return GenerationPlan(segments=[first], warnings=plan.warnings, has_markup=plan.has_markup)

    def _reference_scope(self, body: GenerationCreate,
                         needs_reference: bool) -> tuple[str | None, list[ReferenceAudio]]:
        """Default reference and the set of references whose emotion tags may be used."""
        reference_id = body.reference_id
        profile_id = body.profile_id
        if profile_id is not None:
            from app.services.profile_service import ProfileService

            profiles = ProfileService(self.session, self.settings, self.models.registry)
            profiles.get(profile_id)
            if reference_id is None and needs_reference:
                reference_id = profiles.generation_reference_id(profile_id)
            elif reference_id is not None:
                ref = self.session.get(ReferenceAudio, reference_id)
                if ref is not None and ref.profile_id != profile_id:
                    raise AppError(ErrorCode.VALIDATION_ERROR, status_code=422,
                                   message="La referencia elegida no pertenece al perfil seleccionado.")
        elif reference_id is not None:
            ref = self.session.get(ReferenceAudio, reference_id)
            profile_id = ref.profile_id if ref is not None else None
        if profile_id is None:
            return reference_id, []
        refs = self.session.exec(select(ReferenceAudio).where(ReferenceAudio.profile_id == profile_id)).all()
        return reference_id, list(refs)

    def _tagged_references(self, refs: list[ReferenceAudio], needs_text: bool) -> dict[str, str]:
        """emotion -> best usable reference id (analysed, with the transcript the engine needs)."""
        from app.models.entities import VoiceProfile

        best: dict[str, ReferenceAudio] = {}
        for ref in refs:
            if not ref.emotion_tag or ref.status != ReferenceStatus.ANALYZED:
                continue
            if needs_text and not self._reference_text(ref):
                continue
            profile = self.session.get(VoiceProfile, ref.profile_id) if ref.profile_id else None
            primary = profile is not None and profile.primary_reference_id == ref.id
            current = best.get(ref.emotion_tag)
            score = (primary, ref.quality_score or 0)
            if current is None:
                best[ref.emotion_tag] = ref
            else:
                current_primary = profile is not None and profile.primary_reference_id == current.id
                if score > (current_primary, current.quality_score or 0):
                    best[ref.emotion_tag] = ref
        return {emotion: ref.id for emotion, ref in best.items()}

    def _reference_text(self, ref: ReferenceAudio) -> str:
        has_segment = ref.segment_start_s is not None and ref.segment_end_s is not None
        start, end = (ref.segment_start_s, ref.segment_end_s) if has_segment else (None, None)
        transcript = self.session.exec(select(Transcript).where(
            Transcript.reference_id == ref.id, Transcript.segment_start_s == start,  # noqa: E711
            Transcript.segment_end_s == end)).first()
        return transcript.text.strip() if transcript else ""

    def _resolve_reference(self, reference_id: str | None, needs_text: bool,
                           duration_limits: tuple[float, float] | None) -> tuple[dict, list[str]]:
        if not reference_id:
            raise AppError(ErrorCode.REFERENCE_REQUIRED, status_code=422)
        ref = self.session.get(ReferenceAudio, reference_id)
        if ref is None:
            raise AppError(ErrorCode.NOT_FOUND, status_code=404, message="La referencia no existe.")
        if ref.status != ReferenceStatus.ANALYZED:
            raise AppError(ErrorCode.VALIDATION_ERROR, status_code=422,
                           message="La referencia todavía no está analizada.")

        has_segment = ref.segment_start_s is not None and ref.segment_end_s is not None
        start, end = (ref.segment_start_s, ref.segment_end_s) if has_segment else (None, None)
        text = self._reference_text(ref)
        if needs_text and not text:
            raise AppError(ErrorCode.REFERENCE_TEXT_REQUIRED, status_code=422,
                           message=f"Transcribe el segmento seleccionado de «{ref.original_name}» antes de generar: "
                                   "el modelo necesita el texto exacto de ese fragmento." if has_segment else None)

        duration = (end - start) if has_segment else ((ref.analysis or {}).get("duration_s") or ref.duration_s or 0)
        warnings: list[str] = []
        if duration_limits:
            low, high = duration_limits
            if duration > high + REFERENCE_DURATION_TOLERANCE_S:
                raise AppError(ErrorCode.REFERENCE_TOO_LONG, status_code=422,
                               message=f"La referencia «{ref.original_name}» dura {duration:.1f} s y este modelo "
                                       f"admite hasta {high:.0f} s. Selecciona un segmento (hay sugerencias en su "
                                       "tarjeta).",
                               details={"duracion_s": round(duration, 2), "maximo_s": high})
            if duration < low:
                warnings.append(f"La referencia dura menos de {low:.0f} s; la similitud de voz puede ser baja.")
        return {"reference_id": ref.id, "name": ref.original_name, "sha256": ref.sha256, "start_s": start,
                "end_s": end, "text": text}, warnings

    # ------------------------------------------------------------------ plan / create / variations
    def plan(self, body: GenerationCreate) -> GenerationPlanRead:
        built = self.build(body, check_install=False)
        return GenerationPlanRead(
            segments=[PlannedSegmentRead(index=s.index, text=s.text, emotion=s.emotion, emotion_via=s.emotion_via,
                                         instruction=s.instruction, pause_before_ms=s.pause_before_ms,
                                         pause_after_ms=s.pause_after_ms,
                                         reference_name=(s.reference or {}).get("name"))
                      for s in built.segments],
            warnings=built.warnings, segmented=built.segmented, text_changes=built.text_changes,
            normalize_language=built.normalize_language)

    async def create(self, body: GenerationCreate, kind: str | None = None, parent_id: str | None = None,
                     experiment_id: str | None = None, label: str | None = None,
                     project_id: str | None = None) -> Generation:
        built = self.build(body)
        params = dict(built.params)
        if built.segmented and "seed" in params and params["seed"] is None:
            params["seed"] = random.randint(0, SEED_MAX)  # base seed so per-segment seeds are reproducible
        first_ref = built.segments[0].reference
        gen = Generation(
            engine=built.engine.id, variant=built.variant, text=built.text,
            params=params if built.segmented else built.segments[0].params, seed=params.get("seed"),
            kind=kind or ("preview" if body.preview else "single"), parent_id=parent_id, status=JobStatus.QUEUED,
            experiment_id=experiment_id, label=label, project_id=project_id,
            reference_id=(first_ref or {}).get("reference_id"), profile_id=body.profile_id,
            reference_snapshot=first_ref, warnings=built.warnings,
            expression={"emotion": body.emotion, "intensity": body.intensity, "markup": body.markup,
                        "normalize": body.normalize, "text_changes": built.text_changes},
            postprocess={"config": body.postprocess.model_dump()} if body.postprocess and body.postprocess.is_active
            else None,
        )
        self.session.add(gen)
        self.session.flush()
        if built.segmented:
            for seg in built.segments:
                seg_params = dict(seg.params)
                if "seed" in seg_params and params.get("seed") is not None:
                    seg_params["seed"] = (params["seed"] + seg.index) % (SEED_MAX + 1)
                self.session.add(GenerationSegment(
                    generation_id=gen.id, index=seg.index, text=seg.text, engine=built.engine.id, params=seg_params,
                    seed=seg_params.get("seed"), emotion=seg.emotion, pause_before_ms=seg.pause_before_ms,
                    pause_after_ms=seg.pause_after_ms, instruction=seg.instruction, reference_snapshot=seg.reference))
        self.session.commit()
        self.session.refresh(gen)
        await self.queue.submit(JobSpec(kind=JOB_KIND, payload={"generation_id": gen.id}), job_id=gen.id)
        return gen

    async def create_variations(self, body: GenerationCreate, count: int) -> list[Generation]:
        self.build(body)  # validate once before queuing anything
        seeds = random.sample(range(SEED_MAX), count)
        created: list[Generation] = []
        for seed in seeds:
            variant_body = body.model_copy(update={"params": {**body.params, "seed": seed}})
            parent = created[0].id if created else None
            created.append(await self.create(variant_body, kind="variation", parent_id=parent))
        return created

    async def create_batch(self, body: BatchCreate) -> list[Generation]:
        """Every text × variant × repetition, validated in full before anything is queued."""

        variants = body.variants or [body.variant]
        requests: list[tuple[GenerationCreate, str]] = []
        for text in body.texts:
            for variant in variants:
                for index in range(body.repeat):
                    params = dict(body.params)
                    if body.repeat > 1 or params.get("seed") is None:
                        params["seed"] = random.randint(0, SEED_MAX)
                    label = f"{text[:28]}…" if len(text) > 29 else text
                    if len(variants) > 1:
                        label = f"{label} · {variant}"
                    if body.repeat > 1:
                        label = f"{label} ({index + 1}/{body.repeat})"
                    requests.append((body_of_batch(body, text, variant, params), label))
        if len(requests) > MAX_BATCH:
            raise AppError(ErrorCode.VALIDATION_ERROR, status_code=422,
                           message=f"El lote saldría de {len(requests)} generaciones y el máximo es {MAX_BATCH}. "
                                   "Quita textos, variantes o repeticiones.",
                           details={"total": len(requests), "maximo": MAX_BATCH})
        for index, (request, _) in enumerate(requests):
            try:
                self.build(request)
            except AppError as exc:
                exc.details = {**(exc.details or {}), "elemento": index + 1}
                raise
        return [await self.create(request, label=label) for request, label in requests]

    # ------------------------------------------------------------------ queries
    def get(self, generation_id: str) -> Generation:
        gen = self.session.get(Generation, generation_id)
        if gen is None:
            raise AppError(ErrorCode.GENERATION_NOT_FOUND, status_code=404)
        return gen

    def list(self, limit: int = 50) -> list[Generation]:
        stmt = (select(Generation).where(Generation.experiment_id == None,  # noqa: E711  (own pages)
                                         Generation.project_id == None)  # noqa: E711
                .order_by(Generation.created_at.desc()).limit(limit))  # type: ignore[attr-defined]
        return list(self.session.exec(stmt))

    def segments_of(self, generation_id: str) -> list[GenerationSegment]:
        stmt = select(GenerationSegment).where(GenerationSegment.generation_id == generation_id).order_by(
            GenerationSegment.index)  # type: ignore[arg-type]
        return list(self.session.exec(stmt))

    def audio_path(self, gen: Generation, raw: bool = False) -> Path:
        if gen.status != JobStatus.COMPLETED or not gen.output_path:
            raise AppError(ErrorCode.GENERATION_NOT_FOUND, status_code=404,
                           message="El audio de esta generación no está disponible.")
        if raw:
            return mastering.raw_path(self.settings, gen)
        path = (self.settings.data_dir / gen.output_path).resolve()
        if self.settings.data_dir.resolve() not in path.parents or not path.exists():
            raise AppError(ErrorCode.GENERATION_NOT_FOUND, status_code=404,
                           message="El audio de esta generación no está disponible.")
        return path

    async def delete(self, generation_id: str) -> None:
        gen = self.get(generation_id)
        if not gen.status.is_terminal:
            await self.queue.cancel(gen.id)
        mastering.delete_files(self.settings, gen)
        for seg in self.segments_of(gen.id):
            self.session.delete(seg)
        for child in self.session.exec(select(Generation).where(Generation.parent_id == gen.id)):
            child.parent_id = None
            self.session.add(child)
        self.session.flush()
        self.session.delete(gen)
        self.session.commit()

    # ------------------------------------------------------------------ post-processing
    def _native_speed_parameter(self, gen: Generation) -> str | None:
        try:
            cap = self.models.registry.get(gen.engine).capabilities(gen.variant).controls.get(GenericControl.SPEED)
        except AppError:
            return None
        return cap.parameter if cap and cap.source == ControlSource.NATIVE else None

    def postprocess(self, generation_id: str, config: PostProcessConfig) -> Generation:
        """Re-master a finished generation from its untouched model output (synchronous, seconds at most)."""
        gen = self.get(generation_id)
        if gen.status != JobStatus.COMPLETED:
            raise AppError(ErrorCode.GENERATION_NOT_READY, status_code=409)
        if not config.is_active:
            return self.clear_postprocess(generation_id)
        audio, sr, record = mastering.master(self.settings, gen, self.segments_of(gen.id), config,
                                             self._native_speed_parameter(gen))
        mastering.write_final(self.settings, gen, audio, sr)
        gen.postprocess = record
        gen.updated_at = datetime.now(UTC)
        self.session.add(gen)
        self.session.commit()
        self.session.refresh(gen)
        return gen

    def rate(self, generation_id: str, rating: dict) -> Generation:
        gen = self.get(generation_id)
        cleaned = {k: v for k, v in rating.items() if v not in (None, "")}
        gen.rating = {**cleaned, "rated_at": datetime.now(UTC).isoformat()} if cleaned else None
        self.session.add(gen)
        self.session.commit()
        self.session.refresh(gen)
        return gen

    def clear_postprocess(self, generation_id: str) -> Generation:
        gen = self.get(generation_id)
        if gen.status != JobStatus.COMPLETED:
            raise AppError(ErrorCode.GENERATION_NOT_READY, status_code=409)
        data, sr = sf.read(mastering.raw_path(self.settings, gen), dtype="float32", always_2d=False)
        mastering.write_final(self.settings, gen, np.asarray(data).reshape(-1), sr)
        gen.postprocess = None
        gen.updated_at = datetime.now(UTC)
        self.session.add(gen)
        self.session.commit()
        self.session.refresh(gen)
        return gen

    async def to_read(self, gen: Generation) -> GenerationRead:
        progress, message = (1.0 if gen.status == JobStatus.COMPLETED else 0.0), None
        status = gen.status
        chunks = 0
        try:
            state = await self.queue.get(gen.id)
            status, progress, message, chunks = state.status, state.progress, state.message, state.chunks
        except AppError:  # job no longer in memory (e.g. after restart): DB state is authoritative
            if gen.error_code and gen.status.is_terminal:
                from app.core.errors import MESSAGES

                message = MESSAGES.get(ErrorCode(gen.error_code)) if gen.error_code in ErrorCode.__members__ else None
        if not status.is_terminal and gen.status.is_terminal:  # job finished writing before the queue noticed
            status = gen.status
            progress, message = (1.0 if status == JobStatus.COMPLETED else progress), None
        snap = gen.reference_snapshot
        try:
            progress_available = self.models.registry.get(gen.engine).capabilities(gen.variant).reports_progress
        except AppError:
            progress_available = True
        segments = self.segments_of(gen.id)
        if segments:
            progress_available = True  # segment count gives real progress even for engines without it
        return GenerationRead(
            id=gen.id, kind=gen.kind, engine=gen.engine, variant=gen.variant, text=gen.text, params=gen.params,
            seed=gen.seed, status=status, progress=progress, progress_available=progress_available,
            stream_chunks=chunks, message=message, error_code=gen.error_code,
            reference=GenerationReference(reference_id=snap["reference_id"], name=snap.get("name"),
                                          start_s=snap.get("start_s"), end_s=snap.get("end_s"), text=snap["text"])
            if snap else None,
            duration_s=gen.duration_s,
            audio_url=f"/api/generation/{gen.id}/audio?v={int(gen.updated_at.timestamp())}"
            if gen.status == JobStatus.COMPLETED else None,
            raw_audio_url=f"/api/generation/{gen.id}/audio?version=raw" if gen.status == JobStatus.COMPLETED else None,
            postprocess=GenerationPostProcess.model_validate(gen.postprocess) if gen.postprocess else None,
            experiment_id=gen.experiment_id, label=gen.label, rating=gen.rating,
            evaluation={**gen.evaluation, "stale": gen.evaluation.get("output_path") != gen.output_path}
            if gen.evaluation else None,
            metrics=gen.metrics, warnings=gen.warnings or [], created_at=gen.created_at, updated_at=gen.updated_at,
            expression=gen.expression, parent_id=gen.parent_id,
            segments=[PlannedSegmentRead(index=s.index, text=s.text, emotion=s.emotion,
                                         emotion_via="instruction" if s.instruction and s.emotion else
                                         ("reference" if s.emotion else None),
                                         instruction=s.instruction, pause_before_ms=s.pause_before_ms,
                                         pause_after_ms=s.pause_after_ms,
                                         reference_name=(s.reference_snapshot or {}).get("name"), seed=s.seed,
                                         duration_s=s.duration_s)
                      for s in segments],
        )


# ---------------------------------------------------------------------------
# Background job (runs in the worker thread with its own DB session)
# ---------------------------------------------------------------------------
_STATUS_MESSAGES = {
    JobStatus.LOADING_MODEL: "Cargando modelo…",
    JobStatus.PROCESSING_AUDIO: "Preparando referencia…",
    JobStatus.GENERATING: "Generando voz…",
    JobStatus.POST_PROCESSING: "Guardando audio…",
}


def _reference_input(storage: AudioStorage, snap: dict) -> ReferenceInput:
    return ReferenceInput(audio_path=storage.processed_path(snap["sha256"]), text=snap["text"], sha256=snap["sha256"],
                          start_s=snap.get("start_s"), end_s=snap.get("end_s"))


def _accepts_live_audio(engine: Any, variant: str | None) -> bool:
    """Only engines that declare streaming and whose generate() takes `on_audio` (plugins may predate it)."""
    try:
        if not engine.capabilities(variant).supports_streaming:
            return False
    except AppError:
        return False
    return "on_audio" in inspect.signature(engine.generate).parameters


def run_generation_job(spec: JobSpec, ctx: JobContext) -> dict:
    settings = get_settings()
    models = get_model_manager()
    generation_id = spec.payload["generation_id"]

    with Session(get_engine()) as session:
        gen = session.get(Generation, generation_id)
        if gen is None:
            raise AppError(ErrorCode.GENERATION_NOT_FOUND, status_code=404)
        segments = list(session.exec(select(GenerationSegment).where(
            GenerationSegment.generation_id == gen.id).order_by(GenerationSegment.index)))  # type: ignore[arg-type]

        def phase(status: JobStatus, progress: float, message: str | None = None) -> None:
            ctx.check_cancelled()
            gen.status = status
            gen.updated_at = datetime.now(UTC)
            session.add(gen)
            session.commit()
            ctx.update(status, progress, message or _STATUS_MESSAGES.get(status))

        started = time.perf_counter()
        phase(JobStatus.LOADING_MODEL, 0.02)
        with models.use(gen.engine, gen.variant or "") as engine:
            loaded_at = time.perf_counter()
            storage = AudioStorage(settings.data_dir)
            prepared_cache: dict[str, Any] = {}

            def prepared(snap: dict | None) -> Any:
                if not snap:
                    return None
                if snap["reference_id"] not in prepared_cache:
                    prepared_cache[snap["reference_id"]] = engine.prepare_reference(_reference_input(storage, snap),
                                                                                    gen.variant or "")
                return prepared_cache[snap["reference_id"]]

            stream = StreamWriter(mastering.generation_dir(settings, gen.id), ctx.chunk_ready)
            engine_streams = _accepts_live_audio(engine, gen.variant)

            def generate(request: EngineRequest, progress: Any) -> Any:
                extra = {"on_audio": stream.write} if engine_streams else {}
                return models.run_with_memory_retry(partial(engine.generate, request, progress=progress,
                                                            cancel=ctx.cancel, **extra))

            phase(JobStatus.PROCESSING_AUDIO, 0.1)
            gen_started = time.perf_counter()
            warnings = list(gen.warnings or [])
            details: dict[str, Any] = {}

            if not segments:
                reference = prepared(gen.reference_snapshot)
                phase(JobStatus.GENERATING, 0.15)
                request = EngineRequest(variant=gen.variant or "", text=gen.text, params=gen.params or {},
                                        reference=reference)
                result = generate(request, lambda p, msg: ctx.update(
                    JobStatus.GENERATING, 0.15 + 0.8 * max(0.0, min(1.0, p)), msg))
                audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                sample_rate, seed = result.sample_rate, result.seed
                gen.params = {**(gen.params or {}), **result.effective_params}
                warnings += [w for w in result.warnings if w not in warnings]
                details = dict(result.details)
            else:
                n = len(segments)
                parts: list[SegmentAudio] = []
                for i, seg in enumerate(segments):
                    phase(JobStatus.GENERATING, 0.15 + 0.8 * i / n, f"Generando segmento {i + 1} de {n}…")
                    reference = prepared(seg.reference_snapshot)
                    request = EngineRequest(variant=gen.variant or "", text=seg.text, params=seg.params or {},
                                            reference=reference)
                    result = generate(request, lambda p, msg, i=i: ctx.update(
                        JobStatus.GENERATING, 0.15 + 0.8 * (i + max(0.0, min(1.0, p))) / n,
                        f"Segmento {i + 1} de {n}" + (f": {msg}" if msg else "…")))
                    seg_audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                    seg.seed = result.seed
                    seg.params = {**(seg.params or {}), **result.effective_params}
                    seg.duration_s = round(seg_audio.size / result.sample_rate, 3) if result.sample_rate else None
                    mastering.save_segment(settings, seg, seg_audio, result.sample_rate)
                    session.add(seg)
                    warnings += [w for w in result.warnings if w not in warnings]
                    details.update(result.details)
                    parts.append(SegmentAudio(seg_audio, result.sample_rate, seg.pause_before_ms, seg.pause_after_ms))
                    # live preview: a non-streaming engine shows each finished segment; a streaming one already
                    # sent its audio, so only the pause that follows is added
                    pause = silence(seg.pause_after_ms if i < n - 1 else 0, result.sample_rate)
                    stream.write(pause if engine_streams else np.concatenate([seg_audio, pause]), result.sample_rate)
                session.commit()
                audio, sample_rate = assemble(parts)
                seed = gen.seed
                details["segments"] = n
            gen_elapsed = time.perf_counter() - gen_started

            phase(JobStatus.POST_PROCESSING, 0.96)
            duration = audio.size / sample_rate if sample_rate else 0.0
            mastering.save_raw(settings, gen, audio, sample_rate)
            config = PostProcessConfig.model_validate(gen.postprocess["config"]) if gen.postprocess else None
            final = audio
            if config is not None and config.is_active:
                phase(JobStatus.POST_PROCESSING, 0.97, "Aplicando posprocesado…")
                try:
                    caps_speed = engine.capabilities(gen.variant).controls.get(GenericControl.SPEED)
                    native_speed = (caps_speed.parameter
                                    if caps_speed and caps_speed.source == ControlSource.NATIVE else None)
                    final, sample_rate, gen.postprocess = mastering.master(settings, gen, segments, config,
                                                                           native_speed)
                except AppError:
                    logger.warning("postprocess_failed_in_job", extra={"generation_id": gen.id})
                    warnings.append("No se pudo aplicar el posprocesado; se guardó el audio original del modelo.")
                    gen.postprocess = None
            warnings += [w for w in mastering.write_final(settings, gen, final, sample_rate) if w not in warnings]
            gen.seed = seed
            gen.warnings = warnings
            gen.metrics = {
                **details,
                "model_load_s": round(loaded_at - started, 3),
                "generation_s": round(gen_elapsed, 3),
                "total_s": round(time.perf_counter() - started, 3),
                "rtf": round(gen_elapsed / duration, 3) if duration else None,
                "device": models.loaded.device if models.loaded else None,
                "sample_rate": sample_rate,
            }
            gen.status = JobStatus.COMPLETED
            gen.error_code = None
            gen.updated_at = datetime.now(UTC)
            session.add(gen)
            session.commit()
            logger.info("generation_completed", extra={"generation_id": gen.id, **gen.metrics})
            return {"generation_id": gen.id}


def mark_interrupted_generations() -> int:
    """At startup: generations left queued/running by a previous process will never run. Mark them honestly."""
    with Session(get_engine()) as session:
        stale = list(session.exec(select(Generation).where(Generation.status.in_(  # type: ignore[attr-defined]
            [s for s in JobStatus if not s.is_terminal]))))
        for gen in stale:
            gen.status = JobStatus.FAILED
            gen.error_code = ErrorCode.JOB_INTERRUPTED.value
            gen.updated_at = datetime.now(UTC)
            session.add(gen)
        session.commit()
    if stale:
        logger.warning("generations_interrupted_by_restart", extra={"count": len(stale)})
    return len(stale)


def persist_job_outcome(state: JobState) -> None:
    """Queue hook: reflect failed/cancelled jobs in the Generation row."""
    if state.kind != JOB_KIND or state.status == JobStatus.COMPLETED:
        return
    with Session(get_engine()) as session:
        gen = session.get(Generation, state.job_id)
        if gen is None or gen.status == JobStatus.COMPLETED:
            return
        gen.status = state.status
        gen.error_code = state.error_code
        gen.updated_at = datetime.now(UTC)
        session.add(gen)
        session.commit()


MODEL_LOAD_JOB = "model_load"


def run_model_load_job(spec: JobSpec, ctx: JobContext) -> dict:
    """Preload a model through the GPU queue so it never races with a generation."""
    ctx.update(JobStatus.LOADING_MODEL, 0.1, "Cargando modelo…")
    models = get_model_manager()
    with models.use(spec.payload["engine"], spec.payload["variant"]):
        pass
    return {"engine": spec.payload["engine"], "variant": spec.payload["variant"]}


@lru_cache
def get_job_queue() -> AsyncioJobQueue:
    queue = AsyncioJobQueue()
    queue.register(JOB_KIND, run_generation_job)
    queue.register(MODEL_LOAD_JOB, run_model_load_job)
    queue.on_finish = persist_job_outcome
    return queue
