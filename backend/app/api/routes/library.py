"""/api/library — search, filters, tags, favourites and bulk delete over the «Generar» history."""

from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.api.routes.generation import get_generation_service
from app.core.database import get_session
from app.schemas.library import DeleteMany, DeleteReport, LibraryFacets, LibraryPage, LibraryQuery
from app.services.generation_service import GenerationService
from app.services.library_service import LibraryService

router = APIRouter(prefix="/library", tags=["Biblioteca"])


def get_library_service(session: Session = Depends(get_session),
                        generations: GenerationService = Depends(get_generation_service)) -> LibraryService:
    return LibraryService(session, generations)


@router.get("", response_model=LibraryPage, summary="Buscar en la biblioteca")
def search(query: LibraryQuery = Depends(), service: LibraryService = Depends(get_library_service)) -> LibraryPage:
    return service.search(query)


@router.get("/facets", response_model=LibraryFacets, summary="Etiquetas, motores y voces disponibles")
def facets(service: LibraryService = Depends(get_library_service)) -> LibraryFacets:
    return service.facets()


@router.post("/delete", response_model=DeleteReport, summary="Eliminar varias generaciones")
async def delete_many(body: DeleteMany, service: LibraryService = Depends(get_library_service)) -> DeleteReport:
    return await service.delete_many(body.ids)
