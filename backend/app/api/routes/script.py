"""/api/script — make a script ready for speech without changing its words."""

from dataclasses import asdict

from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.core.database import get_session
from app.schemas.script import ScriptPrepare, ScriptPrepared
from app.services.pronunciation_service import PronunciationService, language_code
from app.text.script import prepare

router = APIRouter(prefix="/script", tags=["Guion"])


@router.post("/prepare", response_model=ScriptPrepared, summary="Preparar un guion para voz (mismas palabras)")
def prepare_script(body: ScriptPrepare, session: Session = Depends(get_session)) -> ScriptPrepared:
    language = language_code(body.language) if body.language else PronunciationService(session).resolve_language(
        body.params, body.profile_id, body.text)
    result = prepare(body.text, language)
    return ScriptPrepared(
        text=result.text, changed=result.text != body.text.strip(), language=language,
        changes=[asdict(c) for c in result.changes], alerts=[asdict(a) for a in result.alerts],
        suggestions=[asdict(s) for s in result.suggestions], stats=asdict(result.stats))
