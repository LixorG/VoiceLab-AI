"""Repeating one sentence of a finished generation: only that sentence changes, the rest is untouched."""

from __future__ import annotations

import time

import pytest
from sqlmodel import Session, select

from app.core.database import get_engine
from app.models.entities import GenerationSegment
from app.models.enums import JobStatus
from app.services.resegment_service import persist_segment_outcome
from app.workers.base import JobState
from tests.audio_fixtures import requires_ffmpeg
from tests.test_best_take import takemock as _takemock  # noqa: F401  (mock engine with per-seed behaviour)
from tests.test_generation import wait_done


@pytest.fixture
def takemock(_takemock, monkeypatch):  # noqa: F811
    """The regeneration job resolves the manager on its own, like the generation job does."""
    from app.services import generation_service, resegment_service

    monkeypatch.setattr(resegment_service, "get_model_manager", generation_service.get_model_manager)
    return _takemock

TEXT = "Primera frase. [pausa:500ms] Segunda frase. [pausa:500ms] Tercera frase."


def segments(generation_id: str) -> list[GenerationSegment]:
    with Session(get_engine()) as session:
        return list(session.exec(select(GenerationSegment).where(
            GenerationSegment.generation_id == generation_id).order_by(GenerationSegment.index)))  # type: ignore[arg-type]


def generated(client, takes: int = 1, seed: int = 100) -> dict:
    body = {"engine": "takemock", "text": TEXT, "params": {"seed": seed}, "takes": takes}
    return wait_done(client, client.post("/api/generation", json=body).json()["job_id"])


def regenerate(client, generation_id: str, index: int, **body) -> dict:
    response = client.post(f"/api/generation/{generation_id}/segments/{index}/regenerate", json=body)
    assert response.status_code == 202, response.text
    return wait_done(client, response.json()["job_id"])


@requires_ffmpeg
def test_only_the_chosen_sentence_is_generated_again(client, takemock, settings):  # noqa: F811
    done = generated(client)
    assert len(done["segments"]) == 3
    before = {s.index: (settings.data_dir / s.audio_path).read_bytes() for s in segments(done["id"])}
    seeds_before = list(takemock.seeds)

    after_done = regenerate(client, done["id"], 1, seed=555)

    assert after_done["status"] == "COMPLETED"
    assert takemock.seeds[len(seeds_before):] == [555]  # one engine call, for that sentence only
    after = {s.index: (settings.data_dir / s.audio_path).read_bytes() for s in segments(done["id"])}
    assert after[0] == before[0] and after[2] == before[2]
    assert after[1] != before[1]
    assert [s.seed for s in segments(done["id"])] == [100, 555, 102]
    assert after_done["audio_url"] != done["audio_url"]  # the joined audio was written again


@requires_ffmpeg
def test_the_corrected_text_replaces_the_sentence(client, takemock, settings):  # noqa: F811
    done = generated(client)
    after = regenerate(client, done["id"], 0, text="Primera frase corregida.")
    assert [s["text"] for s in after["segments"]][0] == "Primera frase corregida."
    assert segments(done["id"])[0].text == "Primera frase corregida."


@requires_ffmpeg
def test_without_a_seed_it_tries_another_reading(client, takemock, settings):  # noqa: F811
    done = generated(client)
    before = segments(done["id"])[2].seed
    regenerate(client, done["id"], 2)
    assert segments(done["id"])[2].seed != before


@requires_ffmpeg
def test_several_takes_keep_the_best_one_and_are_recorded(client, takemock, settings):  # noqa: F811
    done = generated(client)
    after = regenerate(client, done["id"], 1, seed=100, takes=3)  # seed 100 rambles in this mock
    entry = after["metrics"]["regenerated"][-1]
    assert entry["segment"] == 1 and entry["takes"] == 3
    assert entry["best_take"]["chosen"] == 1 and [t["seed"] for t in entry["best_take"]["takes"]] == [100, 101, 102]
    assert entry["seed"] == 101


@requires_ffmpeg
def test_the_evaluation_is_dropped_because_the_audio_changed(client, takemock, settings):  # noqa: F811
    done = generated(client)
    client.post(f"/api/generation/{done['id']}/evaluate")
    assert client.get(f"/api/generation/{done['id']}").json()["evaluation"] is not None
    assert regenerate(client, done["id"], 0)["evaluation"] is None


@requires_ffmpeg
def test_the_post_processing_is_applied_again(client, takemock, settings):  # noqa: F811
    done = generated(client)
    config = {"peak": {"enabled": True, "target_dbfs": -3.0}}
    assert client.post(f"/api/generation/{done['id']}/postprocess", json=config).status_code == 200
    after = regenerate(client, done["id"], 0)
    assert after["postprocess"]["config"]["peak"]["target_dbfs"] == -3.0
    assert after["postprocess"]["report"] is not None


@requires_ffmpeg
def test_a_generation_with_a_single_sentence_says_to_repeat_it_whole(client, takemock, settings):  # noqa: F811
    done = wait_done(client, client.post("/api/generation", json={
        "engine": "takemock", "text": "Una sola frase."}).json()["job_id"])
    response = client.post(f"/api/generation/{done['id']}/segments/0/regenerate", json={})
    assert response.status_code == 409 and "una sola frase" in response.json()["message"]


@requires_ffmpeg
def test_a_sentence_that_does_not_exist_is_a_clear_404(client, takemock, settings):  # noqa: F811
    done = generated(client)
    response = client.post(f"/api/generation/{done['id']}/segments/9/regenerate", json={})
    assert response.status_code == 404 and "no existe" in response.json()["message"]


@requires_ffmpeg
def test_each_sentence_can_be_listened_to_on_its_own(client, takemock, settings):  # noqa: F811
    done = generated(client)
    response = client.get(f"/api/generation/{done['id']}/segments/1/audio")
    assert response.status_code == 200 and response.content[:4] == b"RIFF"
    assert client.get(f"/api/generation/{done['id']}/segments/7/audio").status_code == 404


def test_empty_text_is_rejected(client, takemock, settings):  # noqa: F811
    response = client.post("/api/generation/x/segments/0/regenerate", json={"text": "   "})
    assert response.status_code == 422


@requires_ffmpeg
def test_a_failed_repeat_keeps_the_audio_that_was_already_there(client, takemock, settings):  # noqa: F811
    done = generated(client)
    persist_segment_outcome(JobState(job_id=done["id"], kind="segment", status=JobStatus.FAILED,
                                     error_code="ENGINE_ERROR"))
    after = client.get(f"/api/generation/{done['id']}").json()
    assert after["status"] == "COMPLETED" and after["audio_url"]
    assert any("el audio anterior se conserva" in w for w in after["warnings"])


@requires_ffmpeg
def test_it_is_refused_while_the_generation_is_still_running(client, takemock, settings):  # noqa: F811
    accepted = client.post("/api/generation", json={"engine": "takemock", "text": TEXT}).json()
    response = client.post(f"/api/generation/{accepted['job_id']}/segments/0/regenerate", json={})
    if response.status_code == 202:  # it had already finished on this machine
        pytest.skip("la generación terminó antes de poder comprobarlo")
    assert response.status_code == 409
    wait_done(client, accepted["job_id"])


@requires_ffmpeg
def test_the_generation_reports_progress_while_the_sentence_is_repeated(client, takemock, settings):  # noqa: F811
    done = generated(client)
    accepted = client.post(f"/api/generation/{done['id']}/segments/0/regenerate", json={}).json()
    assert accepted["generation"]["status"] in ("QUEUED", "LOADING_MODEL", "GENERATING", "COMPLETED")
    deadline = time.time() + 30
    while time.time() < deadline and client.get(f"/api/generation/{done['id']}").json()["status"] != "COMPLETED":
        time.sleep(0.02)
    assert client.get(f"/api/generation/{done['id']}").json()["status"] == "COMPLETED"
