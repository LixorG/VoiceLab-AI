"""Generating from the terminal: argument handling, the voice lookup and a full run with the simulated engine."""

from __future__ import annotations

import argparse
import json

import pytest
from sqlmodel import Session, select

from app.cli import main
from app.cli_speak import (
    _fold,
    _output_path,
    _parse_params,
    _prepare,
    _resolve_voice,
    _settings_for,
    _text_of,
)
from app.core.database import get_engine
from app.core.errors import AppError
from app.models.entities import Generation, VoiceProfile
from tests.audio_fixtures import requires_ffmpeg


@pytest.fixture
def cli(settings, monkeypatch):
    """The CLI prepares everything itself; it only needs the simulated engine and the test settings."""
    monkeypatch.setenv("ENABLE_MOCK_ENGINE", "true")
    from app.core.config import get_settings
    from app.engines.manager import get_model_manager
    from app.engines.registry import get_engine_registry
    from app.services.generation_service import get_job_queue

    # every accessor the CLI uses is cached per process; other tests leave theirs built without the mock engine
    for cached in (get_settings, get_engine_registry, get_model_manager, get_job_queue):
        cached.cache_clear()
    _prepare()  # settings, logging and database, as every command does before running
    yield get_settings()
    for cached in (get_engine_registry, get_model_manager, get_job_queue):
        cached.cache_clear()


def run(*argv: str) -> int:
    return main(list(argv))


def test_params_are_read_as_numbers_booleans_or_text():
    assert _parse_params(["speed=1.1", "nfe_steps=32", "quiet=true", "language=Spanish", "offset=-2"]) == {
        "speed": 1.1, "nfe_steps": 32, "quiet": True, "language": "Spanish", "offset": -2}
    with pytest.raises(ValueError, match="nombre=valor"):
        _parse_params(["speed"])


def test_names_are_compared_the_way_a_person_would():
    assert _fold("  Hanna MÜLLER ") == "hanna muller"


def test_the_text_comes_from_the_argument_or_a_file(tmp_path):
    script = tmp_path / "guion.txt"
    script.write_text("  Hola desde un archivo.\n", encoding="utf-8")
    args = argparse.Namespace(text=None, file=str(script))
    assert _text_of(args) == "Hola desde un archivo."
    assert _text_of(argparse.Namespace(text=" Hola ", file=None)) == "Hola"
    with pytest.raises(ValueError, match="Escribe el texto"):
        _text_of(argparse.Namespace(text="   ", file=None))
    with pytest.raises(ValueError, match="No existe el archivo"):
        _text_of(argparse.Namespace(text=None, file=str(tmp_path / "no.txt")))


def test_the_output_file_keeps_the_name_you_gave_it(tmp_path):
    gen = Generation(engine="mock", text="x", status="QUEUED")
    assert _output_path(argparse.Namespace(out=str(tmp_path / "salida.wav")), gen, "wav").name == "salida.wav"
    assert _output_path(argparse.Namespace(out=str(tmp_path / "salida")), gen, "mp3").name == "salida.mp3"
    assert _output_path(argparse.Namespace(out=None), gen, "wav").name.startswith("voicelab_mock_")


def test_the_settings_saved_for_that_voice_and_engine_are_used():
    profile = VoiceProfile(name="V", slug="v", recommended_settings={
        "qwen3tts": {"variant": "base-0.6b", "params": {"speed": 1.2}}})
    assert _settings_for(profile, "qwen3tts") == ("base-0.6b", {"speed": 1.2})
    assert _settings_for(profile, "f5tts") == (None, {})
    assert _settings_for(None, "f5tts") == (None, {})


def test_a_voice_is_found_by_name_slug_or_a_piece_of_it(cli):
    with Session(get_engine()) as session:
        session.add(VoiceProfile(name="Hanna Müller", slug="hanna-muller"))
        session.add(VoiceProfile(name="Otra voz", slug="otra-voz"))
        session.commit()

        assert _resolve_voice(session, "hanna müller").slug == "hanna-muller"
        assert _resolve_voice(session, "HANNA MULLER").slug == "hanna-muller"  # accents and capitals do not matter
        assert _resolve_voice(session, "otra-voz").slug == "otra-voz"
        assert _resolve_voice(session, "hanna").slug == "hanna-muller"  # only one voice contains it
        assert _resolve_voice(session, None) is None
        with pytest.raises(AppError, match="Hanna Müller, Otra voz"):  # the message lists what there is
            _resolve_voice(session, "Pedro")


def test_the_voices_and_engines_can_be_listed(cli, capsys):
    with Session(get_engine()) as session:
        session.add(VoiceProfile(name="Hanna Miller", slug="hanna-miller", language="Inglés",
                                 default_engine="qwen3tts"))
        session.commit()
    assert run("voices") == 0
    listed = capsys.readouterr().out
    assert "Hanna Miller" in listed and "hanna-miller" in listed and "0 grabaciones" in listed

    assert run("voices", "--json") == 0
    assert json.loads(capsys.readouterr().out)[0]["motor"] == "qwen3tts"

    assert run("engines", "--json") == 0
    assert any(row["motor"] == "mock" for row in json.loads(capsys.readouterr().out))


def test_it_says_when_there_are_no_voices_yet(cli, capsys):
    assert run("voices") == 0
    assert "No hay ninguna voz todavía" in capsys.readouterr().out


@requires_ffmpeg
def test_it_generates_a_file_and_the_generation_is_in_the_library(cli, tmp_path, capsys):
    target = tmp_path / "salida.wav"
    assert run("speak", "Hola desde la terminal. [pausa:400ms] Segunda frase.", "--engine", "mock",
               "--seed", "77", "--out", str(target)) == 0

    printed = capsys.readouterr().out
    assert target.is_file() and target.stat().st_size > 1000
    assert "Listo:" in printed and "semilla 77" in printed and "100 %" in printed
    with Session(get_engine()) as session:
        gen = session.exec(select(Generation)).one()
        assert gen.engine == "mock" and gen.seed == 77 and gen.status == "COMPLETED"
        assert gen.kind == "single"  # it shows up in the Library like anything else


@requires_ffmpeg
def test_json_output_is_one_line_for_scripts(cli, tmp_path, capsys):
    target = tmp_path / "salida.wav"
    assert run("speak", "Una frase.", "--engine", "mock", "--out", str(target), "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["archivo"] == str(target) and data["motor"] == "mock" and data["duracion_s"] > 0
    assert data["avisos"] == []


@requires_ffmpeg
def test_the_text_can_come_from_a_file_and_the_extension_is_added(cli, tmp_path, capsys):
    script = tmp_path / "guion.txt"
    script.write_text("Texto leído desde un archivo.", encoding="utf-8")
    assert run("speak", "--file", str(script), "--engine", "mock", "--out", str(tmp_path / "sin_extension")) == 0
    assert (tmp_path / "sin_extension.wav").is_file()


def test_a_voice_that_does_not_exist_fails_with_a_clear_message(cli, tmp_path, capsys):
    assert run("speak", "Hola", "--engine", "mock", "--voice", "Nadie", "--out", str(tmp_path / "x.wav")) == 1
    assert "No hay ninguna voz que se llame" in capsys.readouterr().err


def test_an_engine_that_does_not_exist_fails_without_a_traceback(cli, tmp_path, capsys):
    assert run("speak", "Hola", "--engine", "inventado", "--out", str(tmp_path / "x.wav")) == 1
    assert "motor" in capsys.readouterr().err.lower()


def test_a_wrong_parameter_is_explained(cli, tmp_path, capsys):
    assert run("speak", "Hola", "--engine", "mock", "--param", "speed", "--out", str(tmp_path / "x.wav")) == 1
    assert "nombre=valor" in capsys.readouterr().err
