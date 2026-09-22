"""Script preparation: same words always; formatting, ElevenLabs tags, punctuation, paragraphs, alerts, cuts."""

from __future__ import annotations

import pytest

from app.generation.markup import parse
from app.text.script import prepare, same_words, sentences

ELEVENLABS = """# 4 Signs You're Becoming a HIGH-VALUE Man 💪

**First**, you stop explaining yourself [pause] to people who don't listen..
- you protect your time
- you say no without guilt!!!
[excited] That's growth.<break time="1.5s" /> [laughs] Trust me , it works.
[stress on next word] Seriously, check www.example.com/tips and (honestly) do it & share it #growth
This line was wrapped by the editor and
continues here without a break
""" + ("1. cuatro things matter; the first one is consistency, and the second one is patience, because nothing worth "
       "having comes fast, so keep going\n")


def test_elevenlabs_script_keeps_every_word_and_becomes_valid_markup():
    result = prepare(ELEVENLABS, "en")
    assert same_words(ELEVENLABS, result.text)
    parse(result.text)  # no MarkupError: every tag is now SpeechMarkup
    lines = result.text.split("\n")
    assert lines[0] == "4 Signs You're Becoming a HIGH-VALUE Man."
    assert lines[1] == ""  # the paragraph break is kept
    assert "you stop explaining yourself [pausa:500ms] to people who don't listen…" in lines[2]
    assert lines[3:5] == ["you protect your time.", "you say no without guilt!"]  # bullets: one sentence each
    assert lines[5] == "[emoción:entusiasmado] That's growth.[/emoción][pausa:1500ms] [risa] Trust me, it works."
    assert lines[6].startswith("[énfasis] Seriously,[/énfasis] check www.example.com/tips")  # the link is untouched
    assert lines[6].endswith("share it growth.")
    assert lines[7] == "This line was wrapped by the editor and continues here without a break."
    kinds = {c.kind for c in result.changes}
    assert {"formato", "emoji", "etiqueta", "puntuación", "párrafo"} <= kinds
    alerts = {a.kind: a.excerpt for a in result.alerts}
    assert alerts["idioma"] == "cuatro" and alerts["mayúsculas"] == "HIGH, VALUE" and alerts["lista"] == "1."
    assert alerts["enlace"] == "www.example.com/tips" and alerts["paréntesis"] == "(honestly)"
    assert alerts["símbolo"] == "&"


def test_long_sentences_get_a_proposed_cut_that_is_not_applied():
    result = prepare(ELEVENLABS, "en")
    [suggestion] = result.suggestions
    assert suggestion.before in result.text  # not applied
    assert suggestion.after.endswith("patience. Because nothing worth having comes fast, so keep going.")
    assert same_words(suggestion.before, suggestion.after)
    spanish = prepare("Hoy quiero contarte algo que me pasó la semana pasada en el trabajo con mi jefe, y todavía "
                      "no sé muy bien qué pensar de todo lo que dijo aquella tarde.", "es")
    assert spanish.suggestions[0].after.split(". ")[1].startswith("Y todavía")


@pytest.mark.parametrize(("tag", "converted"), [
    ("[whispers] a secret.", "[susurro] a secret.[/susurro]"),
    ("[emphasized] this matters.", "[énfasis] this matters.[/énfasis]"),
    ("[sighs] okay.", "[suspiro] okay."),
    ("[breathes] okay.", "[respira] okay."),
    ("[long pause] okay.", "[pausa:1000ms] okay."),
    ("<break time=\"700ms\"/> okay.", "[pausa:700ms] okay."),
    ("[happily] hello there.", "[emoción:feliz] hello there.[/emoción]"),
    ("[sad] it ended. And then", "[emoción:triste] it ended.[/emoción] And then."),
])
def test_elevenlabs_tags(tag, converted):
    result = prepare(tag, "en")
    assert result.text == converted
    parse(result.text)


def test_voicelab_markup_is_left_alone_and_unknown_tags_are_dropped_with_a_note():
    text = "[pausa:800ms] Hola. [whisper]secreto[/whisper] y [emoción:feliz]bien.[/emoción] [sarcastic] Claro."
    result = prepare(text, "es")
    assert result.text == "[pausa:800ms] Hola. [whisper]secreto[/whisper] y [emoción:feliz]bien.[/emoción] Claro."
    assert any(a.kind == "etiqueta" and a.excerpt == "[sarcastic]" for a in result.alerts)
    assert prepare("Todo [/emoción] listo.", "es").text == "Todo [/emoción] listo."


def test_punctuation_and_spacing():
    result = prepare("Hola , qué tal?? Bien!!Gracias.Adiós...\nUn 50 % más, a las 10:30 y 3.5 kg.", "es")
    assert result.text == "Hola, qué tal? Bien! Gracias. Adiós…\nUn 50 % más, a las 10:30 y 3.5 kg."


def test_markdown_emojis_and_hashtags():
    result = prepare("## Título\n*cursiva* y __negrita__ con `código` 🎉 #viral", "es")
    assert result.text == "Título.\ncursiva y negrita con código viral."


def test_stats_and_length_alerts():
    short = prepare("Hola a todos.", "es")
    assert short.stats.words == 3 and short.stats.sentences == 1 and short.stats.paragraphs == 1
    assert any(a.kind == "longitud" for a in short.alerts)
    long_text = ("Esta es una frase de prueba con diez palabras en total. " * 30).strip()
    result = prepare(long_text, "es")
    assert result.stats.words == 330 and result.stats.average_words == 11.0 and result.stats.longest_words == 11
    assert result.stats.seconds > 100 and any("atención" in a.message for a in result.alerts)
    assert prepare("Hola.\n\n[pausa:500ms] Adiós.", "es").alerts[-1].kind == "pausa"


def test_spanish_text_with_english_number_words():
    result = prepare("Tengo four ideas para ti.", "es")
    assert [a.excerpt for a in result.alerts if a.kind == "idioma"] == ["four"]


def test_sentences_split_on_punctuation_and_lines():
    assert sentences("Uno. ¿Dos? «Tres». \nCuatro [pausa:1s] cinco…") == ["Uno.", "¿Dos?", "«Tres».", "Cuatro  cinco…"]


def test_api(client):
    res = client.post("/api/script/prepare", json={"text": "hello there!!\n- one idea\n- another idea",
                                                   "params": {"language": "English"}})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["language"] == "en" and body["changed"] is True
    assert body["text"] == "hello there!\none idea.\nanother idea."
    assert body["stats"]["sentences"] == 3 and body["changes"][0]["kind"] == "formato"
    same = client.post("/api/script/prepare", json={"text": "Todo bien.", "language": "Español"}).json()
    assert same["changed"] is False and same["language"] == "es" and same["changes"] == []
    assert client.post("/api/script/prepare", json={"text": ""}).status_code == 422


def test_a_long_single_paragraph_is_reported():
    block = "Una idea que sigue y sigue sin que nadie haga una pausa larga. " * 12
    assert "párrafo" in {a.kind for a in prepare(block, "es").alerts}
    split = block.replace(". Una", ".\n\nUna", 6)
    assert "párrafo" not in {a.kind for a in prepare(split, "es").alerts}
