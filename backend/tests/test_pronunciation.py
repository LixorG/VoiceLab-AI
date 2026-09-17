"""Pronunciation: number/abbreviation normalisation, user dictionary, generation plan and WER comparison."""

from __future__ import annotations

import pytest

from app.evaluation.intelligibility import error_rates
from app.text.normalization import DictionaryEntry, normalize_text, unique_changes
from tests.test_generation import engines as engines  # noqa: F401  (fixture re-export)


@pytest.mark.parametrize(("text", "expected"), [
    ("Quiero el 100% en 2026.", "Quiero el cien por ciento en dos mil veintiséis."),
    ("Tengo 21 años y 1 hermano.", "Tengo veintiún años y un hermano."),
    ("Mido 1,75 m y peso 70,5 kg.", "Mido uno coma setenta y cinco metros y peso setenta coma cinco kilos."),
    ("Cuesta $20.000 o 15 €.", "Cuesta veinte mil pesos o quince euros."),
    ("Pagué USD 12,50.", "Pagué doce dólares con cincuenta centavos."),
    ("A las 10:30 y a las 11:00.", "A las diez y treinta y a las once en punto."),
    ("El Sr. Pérez llegó 1º.", "El señor Pérez llegó primero."),
    ("Frío: -5 °C, etc. Vive aquí.", "Frío: menos cinco grados, etcétera. Vive aquí."),
])
def test_spanish_normalisation(text, expected):
    assert normalize_text(text, "es")[0] == expected


def test_english_and_unsupported_language():
    assert (normalize_text("It costs $19.99, Mr. Lee.", "en")[0]
            == "It costs nineteen dollars and ninety-nine cents, mister Lee.")
    assert normalize_text("J'ai 21 ans.", "fr")[0] == "J'ai 21 ans."  # untouched: only es/en are normalised


def test_dictionary_wins_and_changes_are_listed():
    entries = [DictionaryEntry("IA", "i a"), DictionaryEntry("GPU", "gepeú", case_sensitive=True)]
    text, changes = normalize_text("La IA usa la GPU y la gpu. 3 veces.", "es", entries)
    assert text == "La i a usa la gepeú y la gpu. tres veces."
    kinds = {c.kind for c in changes}
    assert kinds == {"diccionario", "número"}
    # longer terms are applied first, so the change list follows that order
    assert [(c.original, c.replacement) for c in unique_changes(changes)] == [("GPU", "gepeú"), ("IA", "i a"),
                                                                             ("3", "tres")]


def test_numbers_can_be_disabled():
    text, changes = normalize_text("Son 100 €.", "es", [], numbers=False)
    assert text == "Son 100 €." and changes == []


def test_error_rates_compare_numbers_as_words():
    assert error_rates("Cuesta 100 €", "Cuesta cien euros", "es") == (0.0, 0.0)
    assert error_rates("Cuesta 100 €", "Cuesta cien euros")[0] > 0  # without a language, written as-is


def test_dictionary_crud_and_preview(client):
    res = client.post("/api/pronunciation", json={"term": "IA", "replacement": "i a"})
    assert res.status_code == 201, res.text
    entry = res.json()
    assert entry["profile_id"] is None and entry["case_sensitive"] is False

    assert client.post("/api/pronunciation", json={"term": "  ", "replacement": "x"}).status_code == 422
    assert client.post("/api/pronunciation", json={"term": "x", "replacement": "y",
                                                   "profile_id": "nope"}).status_code == 404

    body = {"text": "La IA en 2026.", "language": "es"}
    preview = client.post("/api/pronunciation/preview", json=body).json()
    assert preview["text"] == "La i a en dos mil veintiséis."
    assert preview["numbers_supported"] is True
    assert {c["kind"] for c in preview["changes"]} == {"diccionario", "número"}

    french = client.post("/api/pronunciation/preview", json={"text": "La IA en 2026.", "language": "fr"}).json()
    assert french["text"] == "La i a en 2026." and french["numbers_supported"] is False  # dictionary still applies

    updated = client.put(f"/api/pronunciation/{entry['id']}",
                         json={"term": "IA", "replacement": "inteligencia artificial"}).json()
    assert updated["replacement"] == "inteligencia artificial"
    assert [e["id"] for e in client.get("/api/pronunciation").json()] == [entry["id"]]

    assert client.delete(f"/api/pronunciation/{entry['id']}").status_code == 204
    assert client.get("/api/pronunciation").json() == []
    assert client.put(f"/api/pronunciation/{entry['id']}",
                      json={"term": "a", "replacement": "b"}).json()["error_code"] == "PRONUNCIATION_NOT_FOUND"


def test_plan_reports_changes_and_toggle(client, engines):  # noqa: F811
    body = {"engine": "mock", "text": "Costó 15 € en 2026.", "params": {}}
    res = client.post("/api/generation/plan", json=body)
    assert res.status_code == 200, res.text
    plan = res.json()
    assert plan["segments"][0]["text"] == "Costó quince euros en dos mil veintiséis."
    assert plan["normalize_language"] == "es"
    assert [c["original"] for c in plan["text_changes"]] == ["15 €", "2026"]

    plain = client.post("/api/generation/plan", json={**body, "normalize": False}).json()
    assert plain["segments"][0]["text"] == "Costó 15 € en 2026." and plain["text_changes"] == []


def test_markup_tags_survive_normalisation(client, engines):  # noqa: F811
    body = {"engine": "mock", "text": "Son 2 cosas. [pausa:300ms] Y 3 más."}
    plan = client.post("/api/generation/plan", json=body).json()
    texts = [s["text"] for s in plan["segments"]]
    assert texts == ["Son dos cosas.", "Y tres más."]
    assert plan["segments"][0]["pause_after_ms"] == 300
