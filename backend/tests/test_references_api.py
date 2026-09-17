import os

import numpy as np
import pytest
import soundfile as sf

from tests.audio_fixtures import requires_ffmpeg, speechlike, transcode, write_wav

pytestmark = requires_ffmpeg


def _upload(client, path, name=None, mime="audio/wav"):
    with path.open("rb") as fh:
        return client.post("/api/references", files={"file": (name or path.name, fh, mime)})


@pytest.fixture
def clean_wav(tmp_path):
    return write_wav(tmp_path / "clean.wav", speechlike(words=20))


def test_upload_wav_runs_pipeline(client, settings, clean_wav):
    res = _upload(client, clean_wav)
    assert res.status_code == 201, res.text
    body = res.json()
    ref = body["reference"]
    assert body["duplicate"] is False
    assert ref["status"] == "ANALYZED"
    assert ref["format"] == "wav" and ref["sample_rate"] == 24000 and ref["channels"] == 1
    assert ref["quality_score"] >= 70 and ref["quality_label"]
    assert "speech_regions" not in ref["analysis"]
    assert ref["urls"]["peaks"].endswith("/peaks")

    originals = list((settings.data_dir / "references" / "original").iterdir())
    assert len(originals) == 1 and not os.access(originals[0], os.W_OK)  # immutable
    processed = next((settings.data_dir / "references" / "processed").iterdir())
    info = sf.info(processed)
    assert (info.samplerate, info.channels, info.subtype) == (24000, 1, "FLOAT")
    assert not list((settings.data_dir / "cache" / "uploads").iterdir())


def test_duplicate_upload_is_not_reprocessed(client, clean_wav):
    first = _upload(client, clean_wav).json()["reference"]
    second = _upload(client, clean_wav, name="copia.wav").json()
    assert second["duplicate"] is True
    assert second["reference"]["id"] == first["id"]
    assert len(client.get("/api/references").json()) == 1


def test_mp3_stereo_44k_is_converted(client, tmp_path):
    src = write_wav(tmp_path / "src.wav", speechlike(sr=44100, words=15), sr=44100, channels=2)
    mp3 = transcode(src, tmp_path / "voz.mp3", "-b:a", "192k")
    ref = _upload(client, mp3, mime="audio/mpeg").json()["reference"]
    assert ref["status"] == "ANALYZED"
    assert (ref["sample_rate"], ref["channels"], ref["format"]) == (44100, 2, "mp3")
    assert ref["analysis"]["sample_rate"] == 24000


@pytest.mark.parametrize("ext,args", [("flac", []), ("ogg", ["-c:a", "libvorbis"]),
                                      ("m4a", ["-c:a", "aac"]), ("opus", ["-c:a", "libopus"]),
                                      ("aiff", [])])
def test_other_formats_decode(client, clean_wav, tmp_path, ext, args):
    converted = transcode(clean_wav, tmp_path / f"voz.{ext}", *args)
    res = _upload(client, converted, mime="application/octet-stream")
    assert res.status_code == 201, res.text
    assert res.json()["reference"]["status"] == "ANALYZED"


def test_fake_audio_rejected_without_leftovers(client, settings, tmp_path):
    fake = tmp_path / "fake.wav"
    fake.write_text("esto no es audio " * 100)
    res = _upload(client, fake)
    assert res.status_code in (415, 422)
    assert set(res.json()) == {"error_code", "message", "details"}
    assert not list((settings.data_dir / "references" / "original").iterdir())
    assert not list((settings.data_dir / "cache" / "uploads").iterdir())


def test_mismatched_container_rejected(client, clean_wav, tmp_path):
    flac = transcode(clean_wav, tmp_path / "x.flac")
    renamed = tmp_path / "disfrazado.wav"
    renamed.write_bytes(flac.read_bytes())
    res = _upload(client, renamed)
    assert res.status_code == 415
    assert res.json()["error_code"] == "UNSUPPORTED_FORMAT"


def test_disallowed_extension(client, clean_wav):
    res = _upload(client, clean_wav, name="voz.exe")
    assert res.status_code == 415


def test_size_limit(client, settings, tmp_path):
    settings.max_upload_mb = 1
    big = write_wav(tmp_path / "big.wav", speechlike(words=80))  # > 1 MB
    res = _upload(client, big)
    assert res.status_code == 413
    assert res.json()["error_code"] == "FILE_TOO_LARGE"
    assert not list((settings.data_dir / "cache" / "uploads").iterdir())


def test_duration_limit(client, settings, clean_wav):
    settings.max_audio_seconds = 2
    res = _upload(client, clean_wav)
    assert res.status_code == 422
    assert res.json()["error_code"] == "AUDIO_TOO_LONG"


def test_recommendation_prefers_cleaner_reference(client, tmp_path, clean_wav):
    noisy = write_wav(tmp_path / "noisy.wav", speechlike(words=20, noise_db=-26, seed=3))
    clean_id = _upload(client, clean_wav).json()["reference"]["id"]
    _upload(client, noisy)
    refs = client.get("/api/references").json()
    assert [r["id"] for r in refs if r["is_recommended"]] == [clean_id]


def test_peaks_and_file_downloads(client, clean_wav):
    ref = _upload(client, clean_wav).json()["reference"]
    peaks = client.get(f"{ref['urls']['peaks']}?buckets=400").json()
    assert peaks["buckets"] == 400 and 0 < max(peaks["peaks"]) <= 1
    processed = client.get(ref["urls"]["processed"])
    assert processed.status_code == 200 and processed.headers["content-type"] == "audio/wav"
    original = client.get(ref["urls"]["original"])
    assert original.content == clean_wav.read_bytes()
    assert client.get(f"/api/audio/references/{ref['id']}/otro").status_code == 422


def test_segment_selection(client, clean_wav):
    ref = _upload(client, clean_wav).json()["reference"]
    url = f"/api/references/{ref['id']}"
    ok = client.patch(url, json={"segment_start_s": 1.0, "segment_end_s": 6.5, "emotion_tag": "neutral"})
    assert ok.status_code == 200 and ok.json()["segment_end_s"] == 6.5
    assert client.patch(url, json={"segment_start_s": 1.0}).status_code == 422
    assert client.patch(url, json={"segment_start_s": 2.0, "segment_end_s": 2.2}).status_code == 422
    too_far = client.patch(url, json={"segment_start_s": 0, "segment_end_s": 9999})
    assert too_far.status_code == 422 and too_far.json()["error_code"] == "VALIDATION_ERROR"
    cleared = client.patch(url, json={"segment_start_s": None, "segment_end_s": None})
    assert cleared.json()["segment_start_s"] is None


def test_delete_removes_files(client, settings, clean_wav):
    ref = _upload(client, clean_wav).json()["reference"]
    assert client.delete(f"/api/references/{ref['id']}").status_code == 204
    assert client.get(f"/api/references/{ref['id']}").status_code == 404
    for sub in ("original", "processed", "analysis"):
        assert not list((settings.data_dir / "references" / sub).iterdir())


def test_reanalyze_uses_original(client, settings, clean_wav):
    ref = _upload(client, clean_wav).json()["reference"]
    res = client.post(f"/api/references/{ref['id']}/reanalyze")
    assert res.status_code == 200 and res.json()["status"] == "ANALYZED"


def test_analysis_cache_reused(client, settings, clean_wav, monkeypatch):
    first = _upload(client, clean_wav).json()["reference"]
    client.delete(f"/api/references/{first['id']}")  # files removed → next upload recomputes
    import app.audio.pipeline as pipeline

    calls = {"n": 0}
    real = pipeline.analyze

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(pipeline, "analyze", counting)
    ref = _upload(client, clean_wav).json()["reference"]
    assert calls["n"] == 1
    client.post(f"/api/references/{ref['id']}/reanalyze")
    assert calls["n"] == 2  # force=True bypasses cache
    # same file in a different scope (profile) would reuse cache: simulate by clearing DB row only
    from sqlmodel import Session

    from app.core.database import get_engine
    from app.models.entities import ReferenceAudio

    with Session(get_engine()) as s:
        s.delete(s.get(ReferenceAudio, ref["id"]))
        s.commit()
    assert _upload(client, clean_wav).status_code == 201
    assert calls["n"] == 2
    assert np.isfinite(ref["quality_score"])


def test_formats_endpoint(client):
    body = client.get("/api/audio/formats").json()
    assert {"wav", "mp3", "flac", "m4a", "ogg", "opus", "aiff"} <= set(body["extensions"])
    assert body["internal_sample_rate"] == 24000
