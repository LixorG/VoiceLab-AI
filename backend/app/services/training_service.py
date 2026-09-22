"""Voice training: a voice profile's recordings → clips → LoRA fine-tune of Qwen3-TTS → a new engine variant.

The trainer runs in its own process (`app.training.qwen_lora`) so it can be cancelled at any moment and returns all
its GPU memory when it ends; the job that drives it goes through the same serial GPU queue as generations, so the
two never compete for the card. When it finishes, the model is published as a Qwen3-TTS variant (`custom:<slug>`,
through the custom-checkpoint table) and compared with normal cloning on a few recordings kept out of training.
"""

from __future__ import annotations

import json
import logging
import queue as queue_mod
import shutil
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from sqlmodel import Session, select

from app.audio.storage import AudioStorage
from app.core.config import PROJECT_ROOT, Settings, get_settings
from app.core.database import get_engine
from app.core.errors import AppError, ErrorCode
from app.engines.custom import variant_id
from app.engines.manager import ModelManager, get_model_manager
from app.models.entities import CustomCheckpoint, ReferenceAudio, TrainingRun, Transcript, VoiceProfile
from app.models.enums import JobStatus, ReferenceStatus
from app.schemas.training import (
    MIN_MINUTES,
    RECOMMENDED_MINUTES,
    TrainingCreate,
    TrainingDatasetRead,
    TrainingReferenceSummary,
    TrainingRunRead,
)
from app.services.profile_service import slugify
from app.training.dataset import (
    Clip,
    align_edited,
    refine_cuts,
    split_utterances,
    usable,
    words_from_timestamps,
)
from app.workers.asyncio_queue import AsyncioJobQueue, JobContext
from app.workers.base import JobSpec

logger = logging.getLogger("voicelab.training")

TRAINING_JOB = "training"
ENGINE = "qwen3tts"
BASE_REPOS = {"base-1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "base-0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base"}
HELD_OUT = ((30, 4), (15, 2))  # (at least N usable clips, keep K out of training to compare afterwards)
HELD_OUT_SECONDS = (3.0, 12.0)
ACTIVE = ("queued", "preparing", "training", "saving", "evaluating")
STAGE_PROGRESS = {"codes": (0.02, "preparing"), "load": (0.05, "preparing"), "save": (0.9, "saving")}


def models_dir(settings: Settings) -> Path:
    return settings.model_dir / "voices"


# ---------------------------------------------------------------------------------------------------- dataset
class _RefClips:
    def __init__(self, ref: ReferenceAudio, path: Path, clips: list[Clip], discarded: dict[str, int],
                 transcribed: bool, duration_s: float) -> None:
        self.ref, self.path, self.clips, self.discarded = ref, path, clips, discarded
        self.transcribed, self.duration_s = transcribed, duration_s


def _full_transcript(session: Session, ref_id: str) -> Transcript | None:
    rows = session.exec(select(Transcript).where(Transcript.reference_id == ref_id,
                                                 Transcript.segment_start_s == None)  # noqa: E711
                        .order_by(Transcript.updated_at.desc()))  # type: ignore[attr-defined]
    return next((t for t in rows if t.word_timestamps), None)


def collect(session: Session, settings: Settings, profile_id: str, audio: bool = False) -> list[_RefClips]:
    """Usable clips of every analysed recording of the profile (with `audio`, cuts refined on the waveform)."""
    storage = AudioStorage(settings.data_dir)
    refs = session.exec(select(ReferenceAudio).where(ReferenceAudio.profile_id == profile_id)
                        .order_by(ReferenceAudio.created_at))  # type: ignore[arg-type]
    out = []
    for ref in refs:
        if ref.status == ReferenceStatus.FAILED:
            continue
        path = storage.processed_path(ref.sha256)
        duration = float((ref.analysis or {}).get("duration_s") or ref.duration_s or 0.0)
        transcript = _full_transcript(session, ref.id)
        if transcript is None or not path.exists():
            out.append(_RefClips(ref, path, [], {}, False, duration))
            continue
        words = words_from_timestamps(transcript.word_timestamps or [])
        if transcript.asr_text is not None and transcript.text.strip() != transcript.asr_text.strip():
            words = align_edited(words, transcript.text)
        clips = split_utterances(words, duration)
        if audio and clips:
            data, sr = sf.read(str(path), dtype="float32", always_2d=False)
            clips = refine_cuts(clips, np.asarray(data).reshape(-1), sr)
        keep, discarded = [], {}
        for clip in clips:
            reason = usable(clip)
            if reason is None:
                keep.append(clip)
            else:
                discarded[reason] = discarded.get(reason, 0) + 1
        out.append(_RefClips(ref, path, keep, discarded, True, duration))
    return out


def summarize(profile_id: str, groups: list[_RefClips]) -> TrainingDatasetRead:
    usable_s = sum(c.duration_s for g in groups for c in g.clips)
    minutes = usable_s / 60
    missing = [g.ref.id for g in groups if not g.transcribed]
    warnings = []
    if missing:
        warnings.append(f"{len(missing)} grabación(es) sin transcripción completa: transcríbelas para usarlas.")
    if minutes < MIN_MINUTES:
        warnings.append(f"Hacen falta al menos {MIN_MINUTES:.0f} minutos de voz transcrita; ahora hay "
                        f"{minutes:.1f}. Sube o graba más audio de esta voz.")
    elif minutes < RECOMMENDED_MINUTES[0]:
        warnings.append(f"Con {minutes:.1f} minutos se puede entrenar, pero se recomiendan "
                        f"{RECOMMENDED_MINUTES[0]:.0f}–{RECOMMENDED_MINUTES[1]:.0f} para un resultado estable.")
    discarded = sum(n for g in groups for n in g.discarded.values())
    if discarded:
        warnings.append(f"Se descartan {discarded} frase(s) (muy cortas, sin pausas donde cortar o con la "
                        "transcripción dudosa).")
    return TrainingDatasetRead(
        profile_id=profile_id,
        references=[TrainingReferenceSummary(
            reference_id=g.ref.id, name=g.ref.original_name, duration_s=round(g.duration_s, 2),
            transcribed=g.transcribed, clips=len(g.clips), usable_s=round(sum(c.duration_s for c in g.clips), 2),
            discarded=g.discarded) for g in groups],
        clips=sum(len(g.clips) for g in groups), usable_minutes=round(minutes, 2),
        recorded_minutes=round(sum(g.duration_s for g in groups) / 60, 2), missing_transcripts=missing,
        ready=minutes >= MIN_MINUTES, warnings=warnings)


def split_held_out(clips: list[tuple[_RefClips, Clip]]) -> tuple[list, list]:
    """Keep a few clips (spread over the recordings) out of training to compare with normal cloning afterwards."""
    keep = next((k for n, k in HELD_OUT if len(clips) >= n), 0)
    candidates = [i for i, (_, c) in enumerate(clips) if HELD_OUT_SECONDS[0] <= c.duration_s <= HELD_OUT_SECONDS[1]]
    if keep == 0 or len(candidates) < keep:
        return clips, []
    step = len(candidates) / keep
    held = {candidates[int(step * k + step / 2)] for k in range(keep)}
    return [c for i, c in enumerate(clips) if i not in held], [c for i, c in enumerate(clips) if i in held]


def speaker_reference(session: Session, profile: VoiceProfile, clips: list[tuple[_RefClips, Clip]]
                      ) -> tuple[_RefClips, Clip]:
    """The clip whose speaker embedding becomes the voice: a 4–12 s clip of the primary recording if possible,
    otherwise the most confidently transcribed one of that length."""
    fitting = [(g, c) for g, c in clips if 4.0 <= c.duration_s <= 12.0] or clips
    primary = [(g, c) for g, c in fitting if g.ref.id == profile.primary_reference_id]
    return max(primary or fitting, key=lambda gc: (gc[1].probability, gc[1].duration_s))


# ---------------------------------------------------------------------------------------------------- service
class TrainingService:
    def __init__(self, session: Session, settings: Settings, models: ModelManager, queue: AsyncioJobQueue) -> None:
        self.session, self.settings, self.models, self.queue = session, settings, models, queue

    def _profile(self, profile_id: str) -> VoiceProfile:
        profile = self.session.get(VoiceProfile, profile_id)
        if profile is None:
            raise AppError(ErrorCode.PROFILE_NOT_FOUND, status_code=404)
        return profile

    def dataset(self, profile_id: str) -> TrainingDatasetRead:
        self._profile(profile_id)
        return summarize(profile_id, collect(self.session, self.settings, profile_id))

    def get(self, run_id: str) -> TrainingRun:
        run = self.session.get(TrainingRun, run_id)
        if run is None:
            raise AppError(ErrorCode.TRAINING_NOT_FOUND, status_code=404)
        return run

    def list(self, profile_id: str | None = None) -> list[TrainingRunRead]:
        stmt = select(TrainingRun).order_by(TrainingRun.created_at.desc())  # type: ignore[attr-defined]
        if profile_id:
            stmt = stmt.where(TrainingRun.profile_id == profile_id)
        return [self.to_read(run) for run in self.session.exec(stmt)]

    def to_read(self, run: TrainingRun) -> TrainingRunRead:
        profile = self.session.get(VoiceProfile, run.profile_id)
        checkpoint = self.session.get(CustomCheckpoint, run.checkpoint_id) if run.checkpoint_id else None
        return TrainingRunRead(
            id=run.id, profile_id=run.profile_id, profile_name=profile.name if profile else None, engine=run.engine,
            base_variant=run.base_variant, name=run.name, status=run.status,  # type: ignore[arg-type]
            progress=run.progress, message=run.message, params=run.params, dataset=run.dataset, history=run.history,
            evaluation=run.evaluation, checkpoint_id=run.checkpoint_id,
            variant=variant_id(checkpoint.slug) if checkpoint else None, error_code=run.error_code,
            error_detail=run.error_detail, created_at=run.created_at, started_at=run.started_at,
            finished_at=run.finished_at)

    async def start(self, body: TrainingCreate) -> TrainingRunRead:
        profile = self._profile(body.profile_id)
        active = self.session.exec(select(TrainingRun).where(TrainingRun.status.in_(ACTIVE))).first()  # type: ignore[attr-defined]
        if active is not None:
            raise AppError(ErrorCode.TRAINING_BUSY, status_code=409, details={"entrenamiento": active.id})
        engine = self.models.registry.get(ENGINE)
        if not engine.is_installed():
            raise AppError(ErrorCode.MODEL_NOT_INSTALLED, status_code=409,
                           details={"paquetes": list(engine.required_packages)})
        if not engine.weights_installed(body.base_variant):
            raise AppError(ErrorCode.MODEL_NOT_INSTALLED, status_code=409,
                           message="Descarga primero los pesos del modelo base elegido (Modelos → Qwen3-TTS).",
                           details={"variante": body.base_variant})
        if self.models.resolve_device() != "cuda":
            raise AppError(ErrorCode.GPU_NOT_AVAILABLE, status_code=409,
                           message="Entrenar una voz necesita una GPU NVIDIA con CUDA.")
        summary = self.dataset(body.profile_id)
        if not summary.ready:
            raise AppError(ErrorCode.TRAINING_DATA_INSUFFICIENT, status_code=422, message=summary.warnings[-1]
                           if summary.warnings else None,
                           details={"minutos": summary.usable_minutes, "minimo": MIN_MINUTES})
        name = body.name or f"{profile.name} (entrenada)"
        run = TrainingRun(profile_id=profile.id, engine=ENGINE, base_variant=body.base_variant, name=name,
                          params={"epochs": body.epochs, "learning_rate": body.learning_rate, "lora_rank": 16,
                                  "lora_alpha": 16.0, "batch_size": 2, "grad_accumulation": 4},
                          dataset={"clips": summary.clips, "minutes": summary.usable_minutes},
                          message="En cola…")
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)
        await self.queue.submit(JobSpec(kind=TRAINING_JOB, payload={"run_id": run.id}), job_id=run.id)
        return self.to_read(run)

    async def cancel(self, run_id: str) -> TrainingRunRead:
        run = self.get(run_id)
        if run.status in ACTIVE:
            await self.queue.cancel(run_id)
            if run.status == "queued":  # never picked up by the worker: mark it here
                _finish(self.session, run, "cancelled", message="Cancelado.")
        return self.to_read(run)

    def delete(self, run_id: str) -> None:
        run = self.get(run_id)
        if run.status in ACTIVE:
            raise AppError(ErrorCode.TRAINING_BUSY, status_code=409,
                           message="Cancela el entrenamiento antes de borrarlo.")
        remove_model(self.session, self.models, run)
        self.session.delete(run)
        self.session.commit()


def remove_model(session: Session, models: ModelManager, run: TrainingRun) -> None:
    """Delete the trained model (its variant, unloading it if loaded, and its folder) and the run's work files."""
    from app.services.checkpoint_service import CheckpointService

    if run.checkpoint_id and session.get(CustomCheckpoint, run.checkpoint_id) is not None:
        CheckpointService(session, models).delete(run.checkpoint_id)  # also removes the folder
    _remove_folder(run)
    shutil.rmtree(get_settings().data_dir / "training" / run.id, ignore_errors=True)


def _remove_folder(run: TrainingRun) -> None:
    run.checkpoint_id = None
    if run.output_path:
        path = Path(run.output_path)
        if path.is_dir() and models_dir(get_settings()) in path.parents:
            shutil.rmtree(path, ignore_errors=True)


def forget_trained_model(session: Session, checkpoint_id: str) -> None:
    """The variant of a trained voice was deleted (here or from Modelos): remove its folder too."""
    for run in session.exec(select(TrainingRun).where(TrainingRun.checkpoint_id == checkpoint_id)):
        _remove_folder(run)
        run.message = "Modelo borrado."
        session.add(run)
    session.commit()


def _finish(session: Session, run: TrainingRun, status: str, message: str | None = None,
            error_code: str | None = None, detail: str | None = None) -> None:
    run.status, run.message, run.error_code, run.error_detail = status, message, error_code, detail
    run.finished_at = datetime.now(UTC)
    if status == "completed":
        run.progress = 1.0
    run.updated_at = datetime.now(UTC)
    session.add(run)
    session.commit()


# ---------------------------------------------------------------------------------------------------- the job
def _reader(stream: Any, out: queue_mod.Queue) -> None:
    for line in stream:
        out.put(line)
    out.put(None)


def _trainer_command(spec_path: Path) -> list[str]:
    return [sys.executable, "-m", "app.training.qwen_lora", str(spec_path)]


def run_trainer(spec_path: Path, ctx: JobContext, on_event: Any) -> dict:
    """Run the trainer process, forwarding its events; killing it if the job is cancelled."""
    backend_dir = PROJECT_ROOT / "backend"
    proc = subprocess.Popen(_trainer_command(spec_path), cwd=str(backend_dir), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    lines: queue_mod.Queue = queue_mod.Queue()
    threading.Thread(target=_reader, args=(proc.stdout, lines), daemon=True).start()
    stderr: list[str] = []
    threading.Thread(target=lambda: stderr.extend(proc.stderr), daemon=True).start()  # type: ignore[arg-type]
    last: dict = {}
    try:
        while True:
            if ctx.cancel.cancelled:
                proc.kill()
                proc.wait(timeout=30)
                raise AppError(ErrorCode.JOB_CANCELLED, status_code=409)
            try:
                line = lines.get(timeout=0.5)
            except queue_mod.Empty:
                continue
            if line is None:
                break
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            last = event
            on_event(event)
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    if last.get("event") == "error":
        oom = last.get("code") == "OUT_OF_MEMORY"
        raise AppError(ErrorCode.GPU_MEMORY_ERROR if oom else ErrorCode.TRAINING_FAILED, status_code=500,
                       message="La GPU se quedó sin memoria al entrenar. Cierra otros programas que usen la GPU o "
                               "elige el modelo base 0.6B." if oom else None,
                       details={"detalle": last.get("detail")})
    if last.get("event") != "done" or proc.returncode:
        tail = "".join(stderr)[-800:]
        logger.warning("trainer_failed", extra={"returncode": proc.returncode, "stderr": tail})
        raise AppError(ErrorCode.TRAINING_FAILED, status_code=500, details={"detalle": tail[-300:] or None})
    return last


def run_training_job(spec: JobSpec, ctx: JobContext) -> dict:
    settings = get_settings()
    models = get_model_manager()
    with Session(get_engine()) as session:
        run = session.get(TrainingRun, spec.payload["run_id"])
        if run is None:
            raise AppError(ErrorCode.TRAINING_NOT_FOUND, status_code=404)
        try:
            return _train(session, settings, models, run, ctx)
        except AppError as exc:
            cancelled = exc.code is ErrorCode.JOB_CANCELLED
            detail = (exc.details or {}).get("detalle") if isinstance(exc.details, dict) else None
            _finish(session, run, "cancelled" if cancelled else "failed",
                    message="Cancelado." if cancelled else exc.message, error_code=None if cancelled
                    else exc.code.value, detail=detail)
            raise
        except Exception as exc:
            _finish(session, run, "failed", message="El entrenamiento falló.",
                    error_code=ErrorCode.TRAINING_FAILED.value, detail=f"{type(exc).__name__}: {exc}"[:500])
            raise


def _train(session: Session, settings: Settings, models: ModelManager, run: TrainingRun, ctx: JobContext) -> dict:
    from huggingface_hub import snapshot_download

    from app.asr.manager import get_asr_manager

    def update(status: str, progress: float, message: str) -> None:
        run.status, run.progress, run.message = status, round(progress, 4), message
        run.updated_at = datetime.now(UTC)
        session.add(run)
        session.commit()
        ctx.update(JobStatus.GENERATING if status == "training" else JobStatus.PROCESSING_AUDIO, progress, message)

    run.started_at = datetime.now(UTC)
    update("preparing", 0.01, "Preparando las frases de entrenamiento…")
    profile = session.get(VoiceProfile, run.profile_id)
    if profile is None:
        raise AppError(ErrorCode.PROFILE_NOT_FOUND, status_code=404)
    groups = collect(session, settings, run.profile_id, audio=True)
    clips = [(g, c) for g in groups for c in g.clips]
    minutes = sum(c.duration_s for _, c in clips) / 60
    if minutes < MIN_MINUTES:
        raise AppError(ErrorCode.TRAINING_DATA_INSUFFICIENT, status_code=422,
                       details={"minutos": round(minutes, 2), "minimo": MIN_MINUTES})
    train_clips, held_out = split_held_out(clips)
    ref_group, ref_clip = speaker_reference(session, profile, train_clips)
    run_dir = settings.data_dir / "training" / run.id
    run_dir.mkdir(parents=True, exist_ok=True)
    output = models_dir(settings) / f"{slugify(run.name)}-{run.id[:8]}"
    params = run.params or {}

    def clip_spec(g: _RefClips, c: Clip) -> dict:
        return {"audio_path": str(g.path), "text": c.text, "start_s": c.start_s, "end_s": c.end_s}

    trainer_spec = {
        "base_model_path": snapshot_download(repo_id=BASE_REPOS[run.base_variant], local_files_only=True),
        "output_dir": str(output), "speaker_name": "voicelab", "clips": [clip_spec(g, c) for g, c in train_clips],
        "reference": clip_spec(ref_group, ref_clip), "epochs": params.get("epochs", 10),
        "learning_rate": params.get("learning_rate", 1e-4), "batch_size": params.get("batch_size", 2),
        "grad_accumulation": params.get("grad_accumulation", 4), "lora_rank": params.get("lora_rank", 16),
        "lora_alpha": params.get("lora_alpha", 16.0)}
    spec_path = run_dir / "spec.json"
    spec_path.write_text(json.dumps(trainer_spec, ensure_ascii=False, indent=2), encoding="utf-8")
    run.dataset = {**(run.dataset or {}), "clips": len(train_clips), "minutes": round(
        sum(c.duration_s for _, c in train_clips) / 60, 2), "held_out": [clip_spec(g, c) for g, c in held_out],
        "reference": clip_spec(ref_group, ref_clip), "reference_name": ref_group.ref.original_name}
    run.output_path = str(output)
    session.add(run)
    session.commit()

    # the trainer needs the whole card: free the TTS and ASR models this process holds
    models.unload()
    get_asr_manager().try_unload()
    history: list[dict] = []
    last_write = [0.0]

    def on_event(event: dict) -> None:
        kind = event.get("event")
        if kind == "stage" and event.get("stage") in STAGE_PROGRESS:
            progress, status = STAGE_PROGRESS[event["stage"]]
            update(status, progress, event.get("message") or "")
        elif kind == "stage" and event.get("stage") == "train":
            update("training", 0.06, f"Entrenando con {event.get('clips')} frases…")
        elif kind == "epoch":
            history.append({k: event[k] for k in ("epoch", "talker_loss", "predictor_loss") if k in event})
            run.history = list(history)
        elif kind == "progress" and time.monotonic() - last_write[0] > 2:
            last_write[0] = time.monotonic()
            done = event["step"] / max(1, event["total_steps"])
            eta = event.get("eta_s")
            eta_text = f" · quedan ~{max(1, round(eta / 60))} min" if eta else ""
            update("training", 0.06 + 0.84 * done, f"Época {event['epoch']} de {event['epochs']}{eta_text}")

    result = run_trainer(spec_path, ctx, on_event)
    run.history = result.get("history") or history

    checkpoint = CustomCheckpoint(
        engine=ENGINE, name=run.name, slug=_free_slug(session, slugify(run.name)), base_variant=run.base_variant,
        local_path=str(output), languages=None,
        notes=f"Voz entrenada en VoiceLab con {run.dataset['minutes']:.1f} min de «{profile.name}» "
              f"({params.get('epochs', 10)} épocas). No necesita audio de referencia.")
    session.add(checkpoint)
    session.commit()
    session.refresh(checkpoint)
    from app.services.checkpoint_service import refresh_store

    refresh_store(session)
    run.checkpoint_id = checkpoint.id
    session.add(run)
    session.commit()

    update("evaluating", 0.93, "Comparando con la clonación normal…")
    try:
        from app.training.evaluation import compare

        run.evaluation = compare(session, models, run, variant_id(checkpoint.slug), ctx)
    except AppError as exc:
        if exc.code is ErrorCode.JOB_CANCELLED:
            raise
        run.evaluation = {"skipped": exc.message}
    _finish(session, run, "completed", message="Entrenamiento terminado.")
    return {"run_id": run.id, "variant": variant_id(checkpoint.slug)}


def _free_slug(session: Session, slug: str) -> str:
    taken = {row.slug for row in session.exec(select(CustomCheckpoint).where(CustomCheckpoint.engine == ENGINE))}
    candidate, index = slug, 2
    while candidate in taken:
        candidate, index = f"{slug}-{index}", index + 1
    return candidate


def persist_training_outcome(state: Any) -> None:
    """Queue hook: a training job that ended before its handler could record it (e.g. cancelled while queued)."""
    if state.kind != TRAINING_JOB or state.status == JobStatus.COMPLETED:
        return
    with Session(get_engine()) as session:
        run = session.get(TrainingRun, state.job_id)
        if run is None or run.status not in ACTIVE:
            return
        _finish(session, run, "cancelled" if state.status == JobStatus.CANCELLED else "failed",
                message="Cancelado." if state.status == JobStatus.CANCELLED else state.message,
                error_code=None if state.status == JobStatus.CANCELLED else state.error_code)


def mark_interrupted_trainings() -> int:
    """At startup: runs left active by a closed server can never finish; say so."""
    with Session(get_engine()) as session:
        stale = list(session.exec(select(TrainingRun).where(TrainingRun.status.in_(ACTIVE))))  # type: ignore[attr-defined]
        for run in stale:
            _finish(session, run, "failed", message="El servidor se cerró durante el entrenamiento. Vuelve a lanzarlo.",
                    error_code=ErrorCode.JOB_INTERRUPTED.value)
    return len(stale)
