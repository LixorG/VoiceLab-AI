"""Batch generation and queue control: expansion, up-front validation, queue view and bulk cancel."""

from __future__ import annotations

from tests.audio_fixtures import requires_ffmpeg
from tests.test_generation import _add_transcript, wait_done, wait_status  # noqa: F401  (fixture re-export)
from tests.test_generation import engines as engines
from tests.test_generation import reference as reference  # noqa: F401  (fixture re-export)


def _batch(client, **over):
    body = {"engine": "mock", "texts": ["Primera prueba.", "Segunda prueba."], "params": {"seed": 1}, **over}
    return client.post("/api/generation/batch", json=body)


def test_batch_expands_texts_variants_and_repeats(client, engines):  # noqa: F811
    res = _batch(client, repeat=2)
    assert res.status_code == 202, res.text
    body = res.json()
    assert body["total"] == 4  # 2 texts × 1 variant × 2 repeats
    labels = [item["generation"]["label"] for item in body["items"]]
    assert labels == ["Primera prueba. (1/2)", "Primera prueba. (2/2)",
                      "Segunda prueba. (1/2)", "Segunda prueba. (2/2)"]
    seeds = [item["generation"]["seed"] for item in body["items"]]
    assert len(set(seeds)) == 4  # each repetition gets its own seed
    assert [item["queue_position"] for item in body["items"]] == [1, 2, 3, 4]

    for item in body["items"]:
        wait_done(client, item["job_id"])
    history = client.get("/api/generation").json()
    assert len([g for g in history if g["label"]]) == 4  # they show up in the Generate history


def test_batch_over_several_variants(client, engines):  # noqa: F811
    body = _batch(client, texts=["Una sola frase."], variants=["tone", "tone"]).json()
    assert body["total"] == 2
    assert [item["generation"]["label"] for item in body["items"]] == ["Una sola frase. · tone"] * 2
    for item in body["items"]:
        wait_done(client, item["job_id"])


def test_batch_validates_everything_before_queuing(client, engines):  # noqa: F811
    res = _batch(client, texts=["Bien.", "[emoción:inventada]Mal.[/emoción]"])
    assert res.status_code == 422
    assert res.json()["details"]["elemento"] == 2  # says which one failed
    assert client.get("/api/generation").json() == []  # nothing was queued

    assert _batch(client, texts=["   "]).status_code == 422
    too_many = _batch(client, texts=[f"Texto {i}." for i in range(20)], repeat=8)
    assert too_many.status_code == 422 and too_many.json()["details"]["maximo"] == 60


@requires_ffmpeg
def test_queue_view_and_cancel_all(client, engines, reference):  # noqa: F811
    """The first job blocks inside the engine ("LENTO"), so the rest stay visible in the queue."""
    _, _, refmock = engines
    _add_transcript(reference["id"], "texto")
    body = _batch(client, engine="refmock", texts=["LENTO uno.", "LENTO dos.", "LENTO tres."],
                  reference_id=reference["id"]).json()
    running = body["items"][0]["job_id"]
    wait_status(client, running, "GENERATING")

    jobs = client.get("/api/jobs").json()
    assert [j["position"] for j in jobs] == [0, 1, 2]  # running first, then the waiting ones in order
    assert jobs[0]["job_id"] == running and jobs[0]["text"] == "LENTO uno."
    assert all(j["engine"] == "refmock" and j["generation_kind"] == "single" for j in jobs)

    queued_only = client.post("/api/jobs/cancel?only_queued=true").json()
    assert set(queued_only["cancelled"]) == {item["job_id"] for item in body["items"][1:]}
    assert [j["job_id"] for j in client.get("/api/jobs").json()] == [running]  # the running one was left alone

    assert client.post("/api/jobs/cancel").json()["cancelled"] == [running]
    refmock.release.set()
    for item in body["items"]:
        assert wait_done(client, item["job_id"])["status"] == "CANCELLED"
    assert client.get("/api/jobs").json() == []
