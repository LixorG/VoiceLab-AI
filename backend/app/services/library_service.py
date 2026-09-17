"""Library: search, filter, tag, favourite and bulk-delete the standalone generations of «Generar».

Experiment and project generations are excluded here exactly as in `GenerationService.list`: they have their own
pages, and deleting one from a library would break the project that needs it.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from datetime import datetime, time

from sqlalchemy import func
from sqlmodel import Session, col, select

from app.models.entities import Generation, VoiceProfile
from app.schemas.library import (
    DeleteReport,
    FacetOption,
    LibraryFacets,
    LibraryItem,
    LibraryPage,
    LibraryQuery,
    TagCount,
    clean_tags,
)
from app.services.generation_service import GenerationService


def _fold(text: str) -> str:
    """Lowercase without accents, so «canción» also matches «cancion»."""
    return unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()


class LibraryService:
    def __init__(self, session: Session, generations: GenerationService) -> None:
        self.session = session
        self.generations = generations

    # ------------------------------------------------------------------ query
    def _base(self):
        return select(Generation).where(Generation.experiment_id == None,  # noqa: E711  (own pages)
                                        Generation.project_id == None)  # noqa: E711

    def _filtered(self, query: LibraryQuery):
        stmt = self._base()
        if query.engine:
            stmt = stmt.where(Generation.engine == query.engine)
        if query.profile_id:
            stmt = stmt.where(Generation.profile_id == query.profile_id)
        if query.status is not None:
            stmt = stmt.where(Generation.status == query.status)
        if query.kind:
            stmt = stmt.where(Generation.kind == query.kind)
        if query.favorite is not None:
            stmt = stmt.where(Generation.favorite == query.favorite)
        if query.since:
            stmt = stmt.where(Generation.created_at >= datetime.combine(query.since, time.min))
        if query.until:
            stmt = stmt.where(Generation.created_at <= datetime.combine(query.until, time.max))
        return stmt

    def search(self, query: LibraryQuery) -> LibraryPage:
        rows = list(self.session.exec(self._filtered(query).order_by(col(Generation.created_at).desc())))
        # Text and tag are matched in Python: SQLite's LIKE ignores accents inconsistently and tags live in JSON.
        if query.q:
            needle = _fold(query.q.strip())
            rows = [g for g in rows if needle in _fold(g.text)]
        if query.tag:
            tag = query.tag.strip().lower()
            rows = [g for g in rows if tag in (g.tags or [])]
        total = len(rows)
        page = rows[query.offset:query.offset + query.limit]
        names = self._profile_names({g.profile_id for g in page if g.profile_id})
        return LibraryPage(items=[self.item(g, names) for g in page], total=total, limit=query.limit,
                           offset=query.offset)

    def _profile_names(self, ids: set[str]) -> dict[str, str]:
        if not ids:
            return {}
        rows = self.session.exec(select(VoiceProfile).where(col(VoiceProfile.id).in_(ids)))
        return {p.id: p.name for p in rows}

    def item(self, gen: Generation, profile_names: dict[str, str] | None = None) -> LibraryItem:
        names = profile_names if profile_names is not None else self._profile_names({gen.profile_id or ""} - {""})
        return LibraryItem(
            id=gen.id, kind=gen.kind, engine=gen.engine, variant=gen.variant, text=gen.text, status=gen.status,
            seed=gen.seed, duration_s=gen.duration_s,
            audio_url=f"/api/generation/{gen.id}/audio?v={int(gen.updated_at.timestamp())}"
            if gen.output_path else None,
            favorite=bool(gen.favorite), tags=list(gen.tags or []), rating=gen.rating, profile_id=gen.profile_id,
            profile_name=names.get(gen.profile_id or ""), reference_name=(gen.reference_snapshot or {}).get("name"),
            created_at=gen.created_at)

    def facets(self) -> LibraryFacets:
        rows = list(self.session.exec(self._base()))
        tags = Counter(tag for gen in rows for tag in (gen.tags or []))
        engines = Counter(gen.engine for gen in rows)
        profiles = Counter(gen.profile_id for gen in rows if gen.profile_id)
        names = self._profile_names(set(profiles))
        return LibraryFacets(
            tags=[TagCount(tag=t, count=c) for t, c in sorted(tags.items())],
            engines=[FacetOption(id=e, label=e, count=c) for e, c in sorted(engines.items())],
            profiles=[FacetOption(id=p, label=names.get(p, p), count=c) for p, c in
                      sorted(profiles.items(), key=lambda kv: names.get(kv[0], kv[0]))],
            total=len(rows), favorites=sum(1 for gen in rows if gen.favorite))

    # ------------------------------------------------------------------ mutations
    def set_favorite(self, generation_id: str, favorite: bool) -> LibraryItem:
        gen = self.generations.get(generation_id)
        gen.favorite = favorite
        return self._save(gen)

    def set_tags(self, generation_id: str, tags: list[str]) -> LibraryItem:
        gen = self.generations.get(generation_id)
        cleaned = clean_tags(tags)
        gen.tags = cleaned or None
        return self._save(gen)

    def _save(self, gen: Generation) -> LibraryItem:
        self.session.add(gen)
        self.session.commit()
        self.session.refresh(gen)
        return self.item(gen)

    async def delete_many(self, ids: list[str]) -> DeleteReport:
        """Deletes one by one so a missing or busy generation does not cancel the rest."""
        deleted, failed = [], []
        for generation_id in dict.fromkeys(ids):
            try:
                await self.generations.delete(generation_id)
                deleted.append(generation_id)
            except Exception:  # noqa: BLE001  (reported to the user as a partial result)
                failed.append(generation_id)
        return DeleteReport(deleted=deleted, failed=failed)

    def count(self) -> int:
        return int(self.session.exec(select(func.count()).select_from(self._base().subquery())).one())
