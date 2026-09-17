"""Experiments: one text + one voice generated with several engines/settings, then compared side by side."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlmodel import Session, select

from app.core.errors import AppError, ErrorCode
from app.models.entities import Experiment, Generation
from app.models.enums import JobStatus
from app.schemas.experiment import (
    ExperimentArm,
    ExperimentCreate,
    ExperimentRead,
    ExperimentSummary,
    ExperimentUpdate,
)
from app.schemas.generation import GenerationCreate
from app.services.generation_service import GenerationService


def _tag_error(err: AppError, engine: str, index: int) -> None:
    """Say which arm failed, keeping the original details (dict or list of parameter errors)."""
    extra = {"motor": engine, "posicion": index + 1}
    if isinstance(err.details, dict):
        err.details = {**err.details, **extra}
    else:
        err.details = {**extra, **({"errores": err.details} if err.details is not None else {})}


class ExperimentService:
    def __init__(self, session: Session, generations: GenerationService) -> None:
        self.session = session
        self.generations = generations

    # ------------------------------------------------------------------ helpers
    def get(self, experiment_id: str) -> Experiment:
        exp = self.session.get(Experiment, experiment_id)
        if exp is None:
            raise AppError(ErrorCode.EXPERIMENT_NOT_FOUND, status_code=404)
        return exp

    def generations_of(self, experiment_id: str) -> list[Generation]:
        stmt = select(Generation).where(Generation.experiment_id == experiment_id).order_by(
            Generation.created_at)  # type: ignore[arg-type]
        return list(self.session.exec(stmt))

    def _body(self, exp: Experiment, arm: ExperimentArm) -> GenerationCreate:
        settings = exp.settings or {}
        return GenerationCreate(engine=arm.engine, variant=arm.variant, text=exp.text, params=arm.params,
                                reference_id=exp.reference_id, profile_id=exp.profile_id,
                                emotion=settings.get("emotion"), intensity=settings.get("intensity", 50),
                                markup=settings.get("markup", True), postprocess=settings.get("postprocess"))

    def _label(self, arm: ExperimentArm, taken: list[str]) -> str:
        if arm.label and arm.label.strip():
            base = arm.label.strip()
        else:
            engine = self.generations.models.registry.get(arm.engine)
            variant = arm.variant or engine.default_variant()
            variants = engine.variants()
            base = engine.display_name
            if len(variants) > 1:
                base += f" · {next((v.label for v in variants if v.id == variant), variant)}"
        label, n = base, 2
        while label in taken:
            label, n = f"{base} ({n})", n + 1
        return label

    async def _queue_arms(self, exp: Experiment, arms: list[ExperimentArm]) -> None:
        bodies = [self._body(exp, arm) for arm in arms]
        for i, body in enumerate(bodies):  # validate everything before queuing anything
            try:
                self.generations.build(body)
            except AppError as err:
                _tag_error(err, body.engine, i)
                raise
        taken = [g.label or "" for g in self.generations_of(exp.id)]
        for arm, body in zip(arms, bodies, strict=True):
            label = self._label(arm, taken)
            taken.append(label)
            await self.generations.create(body, kind="experiment", experiment_id=exp.id, label=label)

    # ------------------------------------------------------------------ commands
    async def create(self, body: ExperimentCreate) -> Experiment:
        exp = Experiment(
            name=(body.name or "").strip() or f"Experimento {datetime.now():%d/%m %H:%M}",
            text=body.text, profile_id=body.profile_id, reference_id=body.reference_id,
            settings={"emotion": body.emotion, "intensity": body.intensity, "markup": body.markup,
                      "postprocess": body.postprocess.model_dump() if body.postprocess else None},
        )
        # Validate against a transient object first so a rejected arm leaves no empty experiment behind.
        for i, arm in enumerate(body.arms):
            try:
                self.generations.build(self._body(exp, arm))
            except AppError as err:
                _tag_error(err, arm.engine, i)
                raise
        self.session.add(exp)
        self.session.commit()
        self.session.refresh(exp)
        await self._queue_arms(exp, body.arms)
        return exp

    async def add_arms(self, experiment_id: str, arms: list[ExperimentArm]) -> Experiment:
        exp = self.get(experiment_id)
        await self._queue_arms(exp, arms)
        self._touch(exp)
        return exp

    def update(self, experiment_id: str, body: ExperimentUpdate) -> Experiment:
        exp = self.get(experiment_id)
        if body.name is not None:
            exp.name = body.name.strip()
        if body.notes is not None:
            exp.notes = body.notes
        self._touch(exp)
        return exp

    async def delete(self, experiment_id: str) -> None:
        exp = self.get(experiment_id)
        for gen in self.generations_of(exp.id):
            await self.generations.delete(gen.id)
        self.session.delete(exp)
        self.session.commit()

    def _touch(self, exp: Experiment) -> None:
        exp.updated_at = datetime.now(UTC)
        self.session.add(exp)
        self.session.commit()
        self.session.refresh(exp)

    # ------------------------------------------------------------------ queries
    def list(self) -> list[ExperimentSummary]:
        stmt = select(Experiment).order_by(Experiment.created_at.desc())  # type: ignore[attr-defined]
        out = []
        for exp in self.session.exec(stmt):
            gens = self.generations_of(exp.id)
            out.append(ExperimentSummary(
                id=exp.id, name=exp.name, text=exp.text, arms=len(gens),
                completed=sum(g.status == JobStatus.COMPLETED for g in gens),
                running=sum(not g.status.is_terminal for g in gens),
                failed=sum(g.status in (JobStatus.FAILED, JobStatus.CANCELLED) for g in gens),
                engines=sorted({g.engine for g in gens}), created_at=exp.created_at, updated_at=exp.updated_at))
        return out

    async def to_read(self, exp: Experiment) -> ExperimentRead:
        return ExperimentRead(
            id=exp.id, name=exp.name, text=exp.text, profile_id=exp.profile_id, reference_id=exp.reference_id,
            notes=exp.notes, settings=exp.settings,
            generations=[await self.generations.to_read(g) for g in self.generations_of(exp.id)],
            created_at=exp.created_at, updated_at=exp.updated_at)
