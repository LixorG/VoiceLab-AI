"""/api/transcription — automatic (faster-whisper) and manual transcripts of references."""

from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.asr.languages import LANGUAGES
from app.asr.manager import ASRManager, ASRStatus, get_asr_manager
from app.core.config import Settings, get_settings
from app.core.database import get_session
from app.schemas.transcription import LanguageOption, TranscribeRequest, TranscriptRead, TranscriptUpdate
from app.services.transcription_service import TranscriptionService, to_read

router = APIRouter(prefix="/transcription", tags=["Transcripción"])


def get_transcription_service(session: Session = Depends(get_session),
                              settings: Settings = Depends(get_settings),
                              asr: ASRManager = Depends(get_asr_manager)) -> TranscriptionService:
    return TranscriptionService(session, settings, asr)


@router.get("/status", response_model=ASRStatus, summary="Estado del modelo de transcripción")
def status(asr: ASRManager = Depends(get_asr_manager)) -> ASRStatus:
    return asr.status()


@router.post("/model/download", response_model=ASRStatus, status_code=202, summary="Descargar el modelo")
def download(asr: ASRManager = Depends(get_asr_manager)) -> ASRStatus:
    return asr.start_download()


@router.post("/model/unload", response_model=ASRStatus, summary="Liberar el modelo de la memoria")
def unload(asr: ASRManager = Depends(get_asr_manager)) -> ASRStatus:
    asr.unload()
    return asr.status()


@router.get("/languages", response_model=list[LanguageOption], summary="Idiomas disponibles")
def languages() -> list[LanguageOption]:
    return [LanguageOption(code=code, name=name) for code, name in LANGUAGES.items()]


@router.post("/references/{reference_id}", response_model=TranscriptRead,
             summary="Transcribir una referencia (completa o su segmento)")
async def transcribe(reference_id: str, body: TranscribeRequest,
                     service: TranscriptionService = Depends(get_transcription_service)) -> TranscriptRead:
    tr = await service.transcribe(reference_id, body.scope, body.language, body.force)
    return to_read(tr, include_words=True)


@router.get("/references/{reference_id}", response_model=list[TranscriptRead],
            summary="Transcripciones de una referencia")
def list_transcripts(reference_id: str,
                     service: TranscriptionService = Depends(get_transcription_service)) -> list[TranscriptRead]:
    return [to_read(tr, include_words=True) for tr in service.list_for(reference_id)]


@router.patch("/transcripts/{transcript_id}", response_model=TranscriptRead, summary="Editar transcripción")
def edit(transcript_id: str, body: TranscriptUpdate,
         service: TranscriptionService = Depends(get_transcription_service)) -> TranscriptRead:
    return to_read(service.update_text(transcript_id, body.text))


@router.post("/transcripts/{transcript_id}/revert", response_model=TranscriptRead,
             summary="Restaurar el texto reconocido automáticamente")
def revert(transcript_id: str,
           service: TranscriptionService = Depends(get_transcription_service)) -> TranscriptRead:
    return to_read(service.revert(transcript_id))
