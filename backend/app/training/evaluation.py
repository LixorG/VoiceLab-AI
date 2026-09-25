"""After training: the trained voice vs normal cloning (Base + reference) on recordings kept out of training.

Both systems read the same held-out texts with the same seeds; each result is compared with the real recordings of
those texts: speaker similarity (WavLM-SV embeddings, if its model is installed) and word error rate (Whisper, if
installed). Automatic estimates, labelled as such; the samples are saved so the user can listen and judge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from sqlmodel import Session

from app.core.config import get_settings
from app.core.errors import AppError, ErrorCode
from app.engines.base import CancelToken, EngineRequest, ReferenceInput
from app.engines.manager import ModelManager
from app.models.entities import TrainingRun, VoiceProfile
from app.training.qwen_lora import ClipSpec, load_clip

SAMPLE_RATE = 24_000
SAME_DIFFERENCE = 0.01  # similarity differences below this are noise


def samples_dir(run_id: str) -> Path:
    return get_settings().data_dir / "training" / run_id / "samples"


def _similarity(encoder: Any, audio: np.ndarray, target: np.ndarray) -> float:
    from app.evaluation.speaker import cosine

    return round(cosine(encoder.embed(audio, SAMPLE_RATE), target), 4)


def compare(session: Session, models: ModelManager, run: TrainingRun, trained_variant: str, ctx: Any) -> dict:
    from app.asr.manager import get_asr_manager
    from app.evaluation.base import EvaluationInput
    from app.evaluation.intelligibility import IntelligibilityEvaluator
    from app.evaluation.speaker import get_speaker_encoder
    from app.services.pronunciation_service import language_code

    held = [ClipSpec(**c) for c in (run.dataset or {}).get("held_out", [])]
    if not held:
        return {"skipped": "Con menos de 15 frases no se reservan grabaciones para comparar: escucha el resultado "
                           "en Generar."}
    encoder = get_speaker_encoder()
    asr = get_asr_manager()
    has_similarity = encoder.model_installed()
    has_wer = asr.backend.package_available() and asr.backend.model_installed()
    profile = session.get(VoiceProfile, run.profile_id)
    language = language_code(profile.language) if profile else None
    reference = ClipSpec(**(run.dataset or {})["reference"])
    cancel: CancelToken = ctx.cancel
    out_dir = samples_dir(run.id)
    out_dir.mkdir(parents=True, exist_ok=True)

    real = [load_clip(c) for c in held]
    for k, audio in enumerate(real):
        sf.write(out_dir / f"{k}_real.wav", audio, SAMPLE_RATE, subtype="PCM_16")
    target = encoder.embed(np.concatenate(real), SAMPLE_RATE) if has_similarity else None

    def clone_input() -> ReferenceInput:
        return ReferenceInput(audio_path=reference.audio_path, text=reference.text,
                              sha256=Path(reference.audio_path).stem, start_s=reference.start_s,
                              end_s=reference.end_s)

    systems: dict[str, list[np.ndarray]] = {}
    with models.use(run.engine, trained_variant) as engine:
        # a fine-tuned F5 still clones from a reference; a trained Qwen voice carries the speaker itself
        prepared = (engine.prepare_reference(clone_input(), trained_variant)
                    if engine.capabilities(trained_variant).requires_reference_audio else None)
        systems["trained"] = [_generate(engine, trained_variant, c.text, k, prepared, cancel)
                              for k, c in enumerate(held)]
    with models.use(run.engine, run.base_variant) as engine:
        prepared = engine.prepare_reference(clone_input(), run.base_variant)
        systems["normal"] = [_generate(engine, run.base_variant, c.text, k, prepared, cancel)
                             for k, c in enumerate(held)]

    wer_eval = IntelligibilityEvaluator(asr) if has_wer else None
    results: dict[str, dict] = {}
    for name, audios in systems.items():
        for k, audio in enumerate(audios):
            sf.write(out_dir / f"{k}_{name}.wav", audio, SAMPLE_RATE, subtype="PCM_16")
        entry: dict[str, Any] = {}
        if target is not None:
            entry["similarity"] = round(float(np.mean([_similarity(encoder, a, target) for a in audios])), 4)
        if wer_eval is not None:
            wers = [wer_eval.evaluate(EvaluationInput(audio=a, sample_rate=SAMPLE_RATE, target_text=c.text,
                                                      language=language)).metrics[0].value
                    for a, c in zip(audios, held, strict=True)]
            entry["wer"] = round(float(np.mean(wers)), 4)
        results[name] = entry
    if target is not None:  # the ceiling: how similar the real recordings are to their own average
        results["real"] = {"similarity": round(float(np.mean([_similarity(encoder, a, target) for a in real])), 4)}

    return {"held_out": len(held), "texts": [c.text for c in held], "systems": results,
            "verdict": _verdict(results), "missing": [m for m, ok in (
                ("similarity", has_similarity), ("wer", has_wer)) if not ok]}


def _generate(engine: Any, variant: str, text: str, k: int, reference: Any, cancel: CancelToken) -> np.ndarray:
    if cancel.cancelled:
        raise AppError(ErrorCode.JOB_CANCELLED, status_code=409)
    params = engine.validate_parameters({"seed": 1000 + k}, variant)
    result = engine.generate(EngineRequest(variant=variant, text=text, params=params, reference=reference),
                             lambda *_: None, cancel)
    audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
    if result.sample_rate != SAMPLE_RATE:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(result.sample_rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, result.sample_rate // g).astype(np.float32)
    return audio


def _verdict(results: dict[str, dict]) -> str:
    trained, normal = results.get("trained", {}), results.get("normal", {})
    parts = []
    if "similarity" in trained and "similarity" in normal:
        diff = trained["similarity"] - normal["similarity"]
        parts.append("se parece más a tu voz real que la clonación normal" if diff > SAME_DIFFERENCE
                     else "se parece menos a tu voz real que la clonación normal" if diff < -SAME_DIFFERENCE
                     else "se parece a tu voz real igual que la clonación normal")
    if "wer" in trained and "wer" in normal:
        diff = trained["wer"] - normal["wer"]
        parts.append("se entiende peor" if diff > 0.03 else "se entiende mejor" if diff < -0.03
                     else "se entiende igual")
    if not parts:
        return "No hay evaluadores automáticos instalados: escucha las muestras para comparar."
    return "La voz entrenada " + " y ".join(parts) + " (estimación automática)."
