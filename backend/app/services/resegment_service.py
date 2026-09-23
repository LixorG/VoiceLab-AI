"""Regenerate one sentence of a finished generation and rebuild its audio.

A long reading usually fails in one place: a word swallowed, a rushed line, a typo. Generating the whole text again
costs everything and changes the rest, so here only the chosen segment is generated again (new seed, corrected text,
or the best of several takes) and the stored ``seg_NNN.wav`` files are joined again with their pauses.

Everything else is left exactly as it was: engine, variant, parameters, reference, emotion and post-processing. It
runs through the same GPU queue as a generation, with the job id of the generation, so progress, cancellation and
the live status in the interface work without a second mechanism.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import UTC, datetime
from functools import partial
from typing import Any

import numpy as np
import soundfile as sf
from sqlmodel import Session, select

from app.audio.storage import AudioStorage
from app.core.config import get_settings
from app.core.database import get_engine
from app.core.errors import AppError, ErrorCode
from app.engines.base import SEED_MAX, ControlSource, EngineRequest, GenericControl
from app.engines.manager import get_model_manager
from app.generation import best_take
from app.generation.assembly import SegmentAudio, assemble, trim_edges
from app.models.entities import Generation, GenerationSegment
from app.models.enums import JobStatus
from app.postprocess.config import PostProcessConfig
from app.schemas.generation import SegmentRegenerate
from app.services import mastering
from app.services.generation_service import (
    GenerationService,
    _reference_input,
    take_scorers,
    take_seeds,
)
from app.workers.asyncio_queue import JobContext
from app.workers.base import JobSpec, JobState

logger = logging.getLogger("voicelab.generation")

SEGMENT_JOB = "segment"


def segment_of(session: Session, generation_id: str, index: int) -> GenerationSegment:
    seg = session.exec(select(GenerationSegment).where(
        GenerationSegment.generation_id == generation_id,
        GenerationSegment.index == index)).first()
    if seg is None:
        raise AppError(ErrorCode.NOT_FOUND, status_code=404, message="Esa frase no existe en esta generación.")
    return seg


class ResegmentService:
    """Queues the regeneration of a single segment (validated synchronously, in Spanish)."""

    def __init__(self, generations: GenerationService) -> None:
        self.generations = generations
        self.session = generations.session
        self.settings = generations.settings

    def segment_audio(self, generation_id: str, index: int):  # noqa: ANN201 (Path)
        seg = segment_of(self.session, generation_id, index)
        path = (self.settings.data_dir / seg.audio_path).resolve() if seg.audio_path else None
        if path is None or not path.exists() or self.settings.data_dir.resolve() not in path.parents:
            raise AppError(ErrorCode.GENERATION_NOT_FOUND, status_code=404,
                           message="El audio de esa frase no está guardado por separado.")
        return path

    async def regenerate(self, generation_id: str, index: int, body: SegmentRegenerate) -> Generation:
        gen = self.generations.get(generation_id)
        if gen.status != JobStatus.COMPLETED:
            raise AppError(ErrorCode.GENERATION_NOT_READY, status_code=409)
        segments = self.generations.segments_of(gen.id)
        if len(segments) < 2:
            raise AppError(ErrorCode.VALIDATION_ERROR, status_code=409,
                           message="Esta generación es una sola frase: vuelve a generarla con «Repetir».")
        if any(s.audio_path is None for s in segments):
            raise AppError(ErrorCode.VALIDATION_ERROR, status_code=409,
                           message="Esta generación no conserva sus frases por separado, así que no se puede "
                                   "rehacer solo una. Vuelve a generarla para poder editarla por frases.")
        seg = segment_of(self.session, gen.id, index)

        status = self.generations.models.status(gen.engine, gen.variant or "")
        if not status.package_installed or not status.weights_installed:
            raise AppError(ErrorCode.MODEL_NOT_INSTALLED, status_code=409,
                           details={"motor": gen.engine, "variante": gen.variant})

        if body.text is not None and body.text != seg.text:
            seg.text = body.text
        params = dict(seg.params or {})
        if "seed" in params:
            params["seed"] = body.seed if body.seed is not None else random.randint(0, SEED_MAX)
            seg.seed = params["seed"]
            seg.params = params
        self.session.add(seg)
        gen.status = JobStatus.QUEUED
        gen.updated_at = datetime.now(UTC)
        self.session.add(gen)
        self.session.commit()
        self.session.refresh(gen)
        await self.generations.queue.submit(
            JobSpec(kind=SEGMENT_JOB, payload={"generation_id": gen.id, "index": index, "takes": body.takes}),
            job_id=gen.id)
        return gen


def _rebuild(settings: Any, gen: Generation, segments: list[GenerationSegment]) -> tuple[np.ndarray, int]:
    """Join the stored segment files again, with the same pauses and the default crossfade."""
    parts: list[SegmentAudio] = []
    for seg in segments:
        data, sr = sf.read(settings.data_dir / seg.audio_path, dtype="float32", always_2d=False)
        parts.append(SegmentAudio(np.asarray(data).reshape(-1), sr, seg.pause_before_ms, seg.pause_after_ms))
    return assemble(parts)


def run_segment_job(spec: JobSpec, ctx: JobContext) -> dict:
    settings = get_settings()
    models = get_model_manager()
    generation_id = spec.payload["generation_id"]
    index = int(spec.payload["index"])
    takes = max(1, min(5, int(spec.payload.get("takes") or 1)))

    with Session(get_engine()) as session:
        gen = session.get(Generation, generation_id)
        if gen is None:
            raise AppError(ErrorCode.GENERATION_NOT_FOUND, status_code=404)
        segments = list(session.exec(select(GenerationSegment).where(
            GenerationSegment.generation_id == gen.id).order_by(GenerationSegment.index)))  # type: ignore[arg-type]
        seg = next(s for s in segments if s.index == index)
        n = len(segments)
        label = f"Frase {index + 1} de {n}"

        def phase(status: JobStatus, progress: float, message: str | None = None) -> None:
            ctx.check_cancelled()
            gen.status = status
            gen.updated_at = datetime.now(UTC)
            session.add(gen)
            session.commit()
            ctx.update(status, progress, message)

        started = time.perf_counter()
        phase(JobStatus.LOADING_MODEL, 0.02, "Cargando modelo…")
        with models.use(gen.engine, gen.variant or "") as engine:
            storage = AudioStorage(settings.data_dir)
            phase(JobStatus.PROCESSING_AUDIO, 0.1, "Preparando referencia…")
            reference = (engine.prepare_reference(_reference_input(storage, seg.reference_snapshot), gen.variant or "")
                         if seg.reference_snapshot else None)
            warnings = list(gen.warnings or [])
            language = (gen.expression or {}).get("language")
            wer_eval, encoder, embedding = take_scorers(gen, settings, warnings) if takes > 1 else (None, None, None)

            phase(JobStatus.GENERATING, 0.15, f"{label}…")
            request = EngineRequest(variant=gen.variant or "", text=seg.text, params=seg.params or {},
                                    reference=reference)
            seeds = take_seeds(seg.params or {}, takes)
            attempts: list[tuple[Any, np.ndarray]] = []
            scores: list[best_take.TakeScore] = []
            for take, seed in enumerate(seeds):
                ctx.check_cancelled()
                params = dict(request.params)
                if seed is not None:
                    params["seed"] = seed
                note = label if len(seeds) == 1 else f"{label} · toma {take + 1} de {len(seeds)}"

                def progress(fraction: float, message: str | None, i: int = take, note: str = note) -> None:
                    ctx.update(JobStatus.GENERATING, 0.15 + 0.75 * (i + max(0.0, min(1.0, fraction))) / len(seeds),
                               note + (f": {message}" if message else "…"))

                result = models.run_with_memory_retry(partial(
                    engine.generate, request.model_copy(update={"params": params}),
                    progress=progress, cancel=ctx.cancel))
                audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                score = best_take.measure(take, result.seed, audio, result.sample_rate)
                if len(seeds) > 1:
                    if wer_eval is not None:
                        best_take.add_wer(score, audio, result.sample_rate, seg.text, language, wer_eval)
                    if encoder is not None and embedding is not None:
                        best_take.add_similarity(score, audio, result.sample_rate, encoder, embedding)
                attempts.append((result, audio))
                scores.append(score)
            if len(scores) == 1:
                result, seg_audio = attempts[0]
                chosen = None
            else:
                winner = best_take.rank(scores, best_take.expected_seconds(seg.text))[0]
                result, seg_audio = attempts[winner.take]
                chosen = best_take.summary(scores, winner, index)

            if engine.capabilities(gen.variant).sentence_chunks:
                pause_before = segments[index - 1].pause_after_ms if index > 0 else seg.pause_before_ms
                seg_audio = trim_edges(seg_audio, result.sample_rate, head=pause_before > 0,
                                       tail=seg.pause_after_ms > 0)
            seg.seed = result.seed
            seg.params = {**(seg.params or {}), **result.effective_params}
            seg.duration_s = round(seg_audio.size / result.sample_rate, 3) if result.sample_rate else None
            mastering.save_segment(settings, seg, seg_audio, result.sample_rate)
            session.add(seg)
            session.commit()

            phase(JobStatus.POST_PROCESSING, 0.92, "Uniendo las frases…")
            audio, sample_rate = _rebuild(settings, gen, segments)
            mastering.save_raw(settings, gen, audio, sample_rate)
            final = audio
            config = PostProcessConfig.model_validate(gen.postprocess["config"]) if gen.postprocess else None
            if config is not None and config.is_active:
                try:
                    caps_speed = engine.capabilities(gen.variant).controls.get(GenericControl.SPEED)
                    native_speed = (caps_speed.parameter
                                    if caps_speed and caps_speed.source == ControlSource.NATIVE else None)
                    final, sample_rate, gen.postprocess = mastering.master(settings, gen, segments, config,
                                                                          native_speed)
                except AppError:
                    logger.warning("postprocess_failed_in_segment_job", extra={"generation_id": gen.id})
                    warnings.append("No se pudo aplicar el posprocesado; se guardó el audio original del modelo.")
                    gen.postprocess = None
            warnings += [w for w in mastering.write_final(settings, gen, final, sample_rate) if w not in warnings]

        history = list((gen.metrics or {}).get("regenerated") or [])
        history.append({"segment": index, "seed": result.seed, "takes": takes,
                        "at": datetime.now(UTC).isoformat(timespec="seconds"),
                        **({"best_take": chosen} if chosen else {})})
        gen.metrics = {**(gen.metrics or {}), "regenerated": history[-20:],
                       "segment_s": round(time.perf_counter() - started, 3)}
        gen.warnings = warnings
        gen.evaluation = None  # the audio changed: the old estimate no longer describes it
        gen.status = JobStatus.COMPLETED
        gen.error_code = None
        gen.updated_at = datetime.now(UTC)
        session.add(gen)
        session.commit()
        logger.info("segment_regenerated", extra={"generation_id": gen.id, "segment": index})
        return {"generation_id": gen.id, "segment": index}


def persist_segment_outcome(state: JobState) -> None:
    """Queue hook: a failed regeneration must not lose the audio that is already there."""
    if state.kind != SEGMENT_JOB or state.status == JobStatus.COMPLETED:
        return
    with Session(get_engine()) as session:
        gen = session.get(Generation, state.job_id)
        if gen is None:
            return
        if gen.output_path:  # the previous audio is intact: the generation is still finished
            gen.status = JobStatus.COMPLETED
            gen.error_code = None
            note = ("Se canceló la repetición de una frase; el audio anterior se conserva."
                    if state.status == JobStatus.CANCELLED else
                    "No se pudo repetir la frase; el audio anterior se conserva.")
            gen.warnings = [*(w for w in (gen.warnings or []) if w != note), note]
        else:
            gen.status = state.status
            gen.error_code = state.error_code
        gen.updated_at = datetime.now(UTC)
        session.add(gen)
        session.commit()
