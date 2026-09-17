from __future__ import annotations

import numpy as np
import pytest

from app.engines.registry import EngineRegistry
from app.generation.assembly import SegmentAudio, assemble
from app.generation.markup import MarkupError, Pause, Sound, Style, TextRun, parse
from app.generation.planner import PlanError, build_instruction, plan_generation, soften_shouting, split_long_text

REG = EngineRegistry().discover()
F5 = REG.get("f5tts").capabilities()
QWEN_CLONE = REG.get("qwen3tts").capabilities("base-1.7b")
QWEN_CV = REG.get("qwen3tts").capabilities("custom-voice-1.7b")


# ---------------------------------------------------------------- parser
def test_plain_text_has_no_markup():
    parsed = parse("Hola... ¿qué tal?")
    assert not parsed.has_markup
    assert parsed.events == [TextRun("Hola... ¿qué tal?", Style())]


def test_pauses_sounds_and_escape():
    parsed = parse("Hola [pausa:500ms] a todos [pausa:1.5s][respira] y [PAUSE:2S] fin \\[literal]")
    kinds = [type(e).__name__ for e in parsed.events]
    assert kinds == ["TextRun", "Pause", "TextRun", "Pause", "Sound", "TextRun", "Pause", "TextRun"]
    assert [e.ms for e in parsed.events if isinstance(e, Pause)] == [500, 1500, 2000]
    assert isinstance(parsed.events[4], Sound) and parsed.events[4].kind == "breath"
    assert parsed.events[-1].text == " fin [literal]"


def test_spans_and_emotions_with_aliases():
    parsed = parse("[emoción:Entusiasmado]¡Esto es [énfasis]increíble[/énfasis]![/emoción] "
                   "[susurro]secreto[/susurro] [tono:grave]grave[/tono] [emotion:happy]happy")
    runs = [e for e in parsed.events if isinstance(e, TextRun)]
    assert runs[0] == TextRun("¡Esto es ", Style(emotion="excited"))
    assert runs[1].style == Style(emotion="excited", emphasis=True)
    assert runs[3] == TextRun(" ", Style())
    assert runs[4].style.whisper and runs[6].style.pitch == "low"
    assert runs[-1].style.emotion == "happy"


@pytest.mark.parametrize(("text", "fragment"), [
    ("hola [pausa]", "duración"),
    ("hola [pausa:abc]", "no válida"),
    ("hola [pausa:20s]", "entre 1 ms"),
    ("hola [emoción:furioso]", "Emoción desconocida"),
    ("hola [cantar]", "Etiqueta desconocida"),
    ("hola [énfasis]sin cerrar", "Falta cerrar"),
    ("hola [/susurro]", "no tiene una etiqueta de apertura"),
    ("[tono:medio]x[/tono]", "Tono no válido"),
])
def test_markup_errors_are_spanish_and_positioned(text, fragment):
    with pytest.raises(MarkupError) as exc:
        parse(text)
    assert fragment in exc.value.message and exc.value.position >= 0


# ---------------------------------------------------------------- planner
def test_f5_merges_unsupported_styles_and_uses_pauses():
    plan = plan_generation(parse("Hola [énfasis]a todos[/énfasis]. [pausa:400ms][respira] Seguimos [risa]."),
                           F5, "F5-TTS", None, 50)
    assert [s.text for s in plan.segments] == ["Hola a todos.", "Seguimos ."]
    assert plan.segments[0].pause_after_ms == 700  # 400 ms + breath 300 ms
    assert not plan.segments[0].instruction
    assert any("énfasis" in w for w in plan.warnings)
    assert any("respira" in w for w in plan.warnings) and any("risa" in w for w in plan.warnings)


def test_segmentation_emotion_needs_tagged_reference():
    text = "Hola. [emoción:feliz]¡Qué bien![/emoción] Adiós."
    without = plan_generation(parse(text), F5, "F5-TTS", None, 50)
    assert len(without.segments) == 1 and any("feliz" in w for w in without.warnings)
    with_ref = plan_generation(parse(text), F5, "F5-TTS", None, 80, reference_emotions={"happy"})
    assert [(s.text, s.emotion, s.emotion_via) for s in with_ref.segments] == [
        ("Hola.", None, None), ("¡Qué bien!", "happy", "reference"), ("Adiós.", None, None)]
    assert any("intensidad" in w for w in with_ref.warnings)
    assert with_ref.segments[1].instruction is None


def test_instruction_engines_get_english_style_instructions():
    plan = plan_generation(parse("[susurro]Ven aquí.[/susurro] [tono:agudo]¡Mira![/tono]"), QWEN_CV,
                           "Qwen3-TTS", "excited", 90)
    assert len(plan.segments) == 2 and plan.warnings == []
    assert plan.segments[0].instruction == "Speak in a very excited tone. Whisper softly."
    assert plan.segments[1].instruction == "Speak in a very excited tone. Use a higher pitch."


def test_qwen_clone_cannot_use_instructions():
    plan = plan_generation(parse("[susurro]hola[/susurro]"), QWEN_CLONE, "Qwen3-TTS", "calm", 50)
    assert plan.segments[0].instruction is None and plan.is_simple
    assert any("susurro" in w for w in plan.warnings)


def test_simple_plan_and_errors():
    assert plan_generation(parse("Solo texto."), F5, "F5-TTS", None, 50).is_simple
    assert not plan_generation(parse("A [pausa:1s] B"), F5, "F5-TTS", None, 50).is_simple
    with pytest.raises(PlanError):
        plan_generation(parse("[pausa:1s] ... [respira]"), F5, "F5-TTS", None, 50)


def test_intensity_words():
    from app.generation.planner import PlannedSegment

    assert build_instruction(PlannedSegment(0, "x", "sad", 10)) == "Speak in a slightly sad tone."
    assert build_instruction(PlannedSegment(0, "x", "friendly", 50)) == "Speak in a warm and friendly tone."
    assert build_instruction(PlannedSegment(0, "x", None, 50)) is None


# ---------------------------------------------------------------- assembly
def test_assembly_inserts_pauses_and_crossfades():
    sr = 1000
    a = SegmentAudio(np.ones(500, dtype=np.float32), sr, pause_before_ms=100, pause_after_ms=200)
    b = SegmentAudio(np.ones(300, dtype=np.float32), sr)
    c = SegmentAudio(np.ones(400, dtype=np.float32), sr, pause_after_ms=50)
    out, out_sr = assemble([a, b, c], crossfade_ms=40)
    assert out_sr == sr
    # 100 lead + 500 + 200 pause + 300 + 400 - 40 crossfade + 50 tail
    assert out.size == 100 + 500 + 200 + 300 + 400 - 40 + 50
    assert np.all(out[:100] == 0) and np.all(out[600:800] == 0)
    assert np.max(out) <= 1.0 + 1e-5
    joined = out[800 + 300 - 40: 800 + 300]
    assert np.allclose(joined, 1.0, atol=1e-5)  # linear crossfade keeps level without overshoot


def test_assembly_resamples_mismatched_segments():
    out, sr = assemble([SegmentAudio(np.ones(2400, dtype=np.float32), 24_000),
                        SegmentAudio(np.ones(1600, dtype=np.float32), 16_000, pause_before_ms=0)])
    assert sr == 24_000 and abs(out.size - (2400 + 2400 - 960)) <= 2


@pytest.mark.parametrize(("text", "expected", "changed"), [
    ("¡CLARO QUE SÍ, YO SÉ QUE PODRÉ LOGRARLOS!", "¡Claro que sí, yo sé que podré lograrlos!", True),
    ("HOLA A TODOS. ESTO ES UNA PRUEBA", "Hola a todos. Esto es una prueba", True),
    ("Dijo: ¿DE VERDAD LO HICISTE? sí", "Dijo: ¿De verdad lo hiciste? sí", True),
    ("ÑANDÚ GRANDE Y ÁGIL corre", "Ñandú grande y ágil corre", True),
    ("El PIB de la UE y la NASA crecen", "El PIB de la UE y la NASA crecen", False),  # acronyms stay
    ("Uso la API REST hoy", "Uso la API REST hoy", False),
])
def test_soften_shouting(text, expected, changed):
    assert soften_shouting(text) == (expected, changed)


def test_plan_lowercases_shouting_with_a_warning():
    plan = plan_generation(parse("Hola. [emoción:feliz]¡CLARO QUE SÍ, LO LOGRARÉ![/emoción]"), F5, "F5-TTS", None, 50,
                           reference_emotions={"happy"})
    assert plan.segments[1].text == "¡Claro que sí, lo lograré!"
    assert any("MAYÚSCULAS" in w for w in plan.warnings)


def test_split_long_text_prefers_sentence_then_clause_boundaries():
    text = "Primera frase corta. Segunda frase algo más larga que la anterior! ¿Tercera? Cuarta frase final."
    parts = split_long_text(text, 45)
    assert all(len(p) <= 45 for p in parts) and " ".join(parts) == text
    assert parts[0] == "Primera frase corta."
    clauses = split_long_text("uno dos tres, cuatro cinco seis, siete ocho nueve, diez once doce", 30)
    assert all(len(p) <= 30 for p in clauses) and clauses[0].endswith(",")
    assert split_long_text("corto", 300) == ["corto"]


def test_long_text_is_split_for_engines_with_a_per_call_limit():
    caps = F5.model_copy(update={"max_chars_per_call": 60})
    text = "Esta es la primera oración del texto largo. " * 4 + "[pausa:500ms] Fin."
    plan = plan_generation(parse(text), caps, "Qwen3-TTS", None, 50)
    assert len(plan.segments) >= 4 and all(len(s.text) <= 60 for s in plan.segments)
    assert [s.pause_after_ms for s in plan.segments][-2:] == [500, 0]  # the user's pause stays at its place
    assert any("se genera por frases" in w for w in plan.warnings)

