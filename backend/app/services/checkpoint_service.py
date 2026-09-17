"""Custom checkpoints: validate what the user declares, store it and publish it as an engine variant."""

from __future__ import annotations

from pathlib import Path

from sqlmodel import Session, select

from app.core.errors import AppError, ErrorCode
from app.engines.custom import CustomVariant, get_custom_variants, variant_id
from app.engines.manager import ModelManager
from app.models.entities import CustomCheckpoint
from app.schemas.checkpoints import CheckpointCreate, CheckpointRead
from app.services.profile_service import slugify

CHECKPOINT_SUFFIXES = (".safetensors", ".pt", ".ckpt")


def refresh_store(session: Session) -> None:
    """Re-publish every stored checkpoint to the in-process store the engines read."""
    rows = session.exec(select(CustomCheckpoint))
    get_custom_variants().replace([
        CustomVariant(id=variant_id(row.slug), engine=row.engine, name=row.name, base_variant=row.base_variant,
                      repo_id=row.repo_id, ckpt_file=row.ckpt_file, local_path=row.local_path,
                      vocab_file=row.vocab_file, vocab_path=row.vocab_path,
                      languages=list(row.languages or []), notes=row.notes)
        for row in rows
    ])


class CheckpointService:
    def __init__(self, session: Session, models: ModelManager) -> None:
        self.session = session
        self.models = models

    # ------------------------------------------------------------------ read
    def list(self, engine_id: str | None = None) -> list[CheckpointRead]:
        stmt = select(CustomCheckpoint).order_by(CustomCheckpoint.name)  # type: ignore[arg-type]
        if engine_id:
            stmt = stmt.where(CustomCheckpoint.engine == engine_id)
        return [self._read(row) for row in self.session.exec(stmt)]

    def get(self, checkpoint_id: str) -> CustomCheckpoint:
        row = self.session.get(CustomCheckpoint, checkpoint_id)
        if row is None:
            raise AppError(ErrorCode.CHECKPOINT_NOT_FOUND, status_code=404)
        return row

    def _read(self, row: CustomCheckpoint) -> CheckpointRead:
        variant = variant_id(row.slug)
        try:
            installed = self.models.registry.get(row.engine).weights_installed(variant)
        except AppError:
            installed = False
        return CheckpointRead(
            id=row.id, engine=row.engine, variant=variant, name=row.name, base_variant=row.base_variant,
            repo_id=row.repo_id, ckpt_file=row.ckpt_file, local_path=row.local_path, vocab_file=row.vocab_file,
            vocab_path=row.vocab_path,
            languages=list(row.languages or []), notes=row.notes, weights_installed=installed,
            created_at=row.created_at)

    # ------------------------------------------------------------------ write
    def create(self, engine_id: str, body: CheckpointCreate) -> CheckpointRead:
        engine = self.models.registry.get(engine_id)
        if not engine.supports_custom_checkpoints:
            raise AppError(ErrorCode.VALIDATION_ERROR, status_code=422,
                           message=f"{engine.display_name} no admite checkpoints propios: solo carga sus variantes "
                                   "oficiales.")
        builtin = {v.id for v in engine.builtin_variants()}  # type: ignore[attr-defined]
        if body.base_variant not in builtin:
            raise AppError(ErrorCode.VALIDATION_ERROR, status_code=422,
                           message="La arquitectura base debe ser una variante oficial del motor.",
                           details={"variantes": sorted(builtin)})
        self._check_paths(body)

        slug = self._free_slug(engine_id, slugify(body.name))
        row = CustomCheckpoint(engine=engine_id, name=body.name, slug=slug, base_variant=body.base_variant,
                               repo_id=body.repo_id, ckpt_file=body.ckpt_file, local_path=body.local_path,
                               vocab_file=body.vocab_file, vocab_path=body.vocab_path,
                               languages=body.languages or None, notes=body.notes)
        self.session.add(row)
        self.session.commit()
        self.session.refresh(row)
        refresh_store(self.session)
        return self._read(row)

    def _check_paths(self, body: CheckpointCreate) -> None:
        if body.local_path:
            path = Path(body.local_path)
            if not path.is_file():
                raise AppError(ErrorCode.VALIDATION_ERROR, status_code=422,
                               message="No encuentro ese archivo de checkpoint en el equipo.",
                               details={"ruta": body.local_path})
            if path.suffix.lower() not in CHECKPOINT_SUFFIXES:
                raise AppError(ErrorCode.VALIDATION_ERROR, status_code=422,
                               message=f"El checkpoint debe ser un archivo {' o '.join(CHECKPOINT_SUFFIXES)}.")
        if body.ckpt_file and Path(body.ckpt_file).suffix.lower() not in CHECKPOINT_SUFFIXES:
            raise AppError(ErrorCode.VALIDATION_ERROR, status_code=422,
                           message=f"El archivo del repositorio debe ser {' o '.join(CHECKPOINT_SUFFIXES)}.")
        if body.vocab_path and not Path(body.vocab_path).is_file():
            raise AppError(ErrorCode.VALIDATION_ERROR, status_code=422,
                           message="No encuentro el archivo de vocabulario en el equipo.",
                           details={"ruta": body.vocab_path})

    def _free_slug(self, engine_id: str, slug: str) -> str:
        taken = {row.slug for row in self.session.exec(select(CustomCheckpoint).where(
            CustomCheckpoint.engine == engine_id))}
        candidate, index = slug, 2
        while candidate in taken:
            candidate, index = f"{slug}-{index}", index + 1
        return candidate

    def delete(self, checkpoint_id: str) -> None:
        row = self.get(checkpoint_id)
        variant = variant_id(row.slug)
        loaded = self.models.loaded
        if loaded is not None and loaded.engine == row.engine and loaded.variant == variant:
            self.models.unload()  # raises MODEL_BUSY while a generation is using it
        self.session.delete(row)
        self.session.commit()
        refresh_store(self.session)
