from pathlib import Path

from sqlalchemy import inspect
from sqlmodel import Session

from app.core.config import PROJECT_ROOT, Settings
from app.core.database import init_engine
from app.models.entities import Generation, GenerationSegment, VoiceProfile
from app.models.enums import JobStatus


def test_relative_paths_resolve_against_project_root(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "./data")
    s = Settings(_env_file=None)
    assert s.data_dir == (PROJECT_ROOT / "data").resolve()
    assert s.max_upload_bytes == 500 * 1024 * 1024


def test_absolute_paths_kept(tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path)
    assert s.data_dir == Path(tmp_path)
    assert s.sqlalchemy_url.startswith("sqlite:///")


def test_all_entities_created_and_roundtrip(tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path)
    s.ensure_directories()
    engine = init_engine(s)
    tables = set(inspect(engine).get_table_names())
    assert {"voiceprofile", "referenceaudio", "transcript", "generation",
            "generationsegment", "model", "preset", "project"} <= tables

    with Session(engine) as session:
        profile = VoiceProfile(
            name="Mi voz", slug="mi-voz", recommended_settings={"f5tts": {"nfe_steps": 32}}
        )
        session.add(profile)
        session.commit()
        gen = Generation(profile_id=profile.id, engine="f5tts", text="Hola", params={"speed": 1.0})
        session.add(gen)
        session.commit()
        session.add(GenerationSegment(generation_id=gen.id, index=0, text="Hola", engine="f5tts"))
        session.commit()
        session.refresh(gen)
        assert gen.status is JobStatus.QUEUED
        assert session.get(VoiceProfile, profile.id).recommended_settings["f5tts"]["nfe_steps"] == 32


def test_job_status_terminal():
    assert JobStatus.COMPLETED.is_terminal and not JobStatus.GENERATING.is_terminal
