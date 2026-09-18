"""Delivery formats (WAV/MP3/OGG/FLAC) and project subtitles (SRT/VTT) timed like the exported audio."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest
import soundfile as sf

from app.audio.export import export_audio
from app.core.errors import AppError
from app.generation.assembly import SegmentAudio, assemble, timeline
from app.generation.subtitles import build_cues, spoken_text, to_srt, to_vtt
from tests.audio_fixtures import requires_ffmpeg
from tests.fakes import tone
from tests.test_advanced_generation import engines as engines  # noqa: F401  (fixture re-export)
from tests.test_projects import _project, _wait_project

SR = 24_000


def test_timeline_matches_assembly_with_pauses_and_crossfades():
    parts = [SegmentAudio(tone(1.0), SR, 0, 500), SegmentAudio(tone(2.0), SR, 0, 0), SegmentAudio(tone(0.5), SR, 0, 0)]
    audio, sr = assemble(parts)
    spans = timeline(parts)
    assert spans[0] == (0.0, 1.0)
    assert spans[1][0] == pytest.approx(1.5)  # after the 500 ms pause
    assert spans[2][0] < spans[1][1]  # no pause: the crossfade makes them overlap slightly
    assert spans[-1][1] == pytest.approx(audio.size / sr, abs=1e-3)  # ends exactly where the audio ends


def test_cues_strip_markup_split_long_segments_and_format():
    assert spoken_text("Hola [pausa:500ms] mundo. [emoción:feliz]¡Qué bien![/emoción]") == "Hola mundo. ¡Qué bien!"
    assert spoken_text("[texto sin cerrar", markup=True) == "[texto sin cerrar"
    assert spoken_text("[pausa:1s] literal", markup=False) == "[pausa:1s] literal"

    long = ("Primera frase bastante larga para el ejemplo de hoy. Segunda frase que también ocupa sitio en pantalla. "
            "Y una tercera para cerrar.")
    cues = build_cues([(0.0, 2.0), (2.5, 10.5)], ["Corta.", long])
    assert cues[0].start_s == 0.0 and cues[0].end_s == 2.0 and cues[0].text == "Corta."
    rest = cues[1:]
    assert len(rest) >= 2 and rest[0].start_s == 2.5 and rest[-1].end_s == 10.5
    assert all(len(line) <= 84 for c in rest for line in c.text.split("\n"))
    assert all(a.end_s == pytest.approx(b.start_s) for a, b in zip(rest, rest[1:], strict=False))  # contiguous

    comma = build_cues([(0.0, 9.0)], ["El segundo párrafo cuesta 15 € y dura un poco más, para que el subtítulo tenga "
                                      "que partirse en varias líneas y comprobar el reparto."])
    assert comma[0].text.replace("\n", " ") == "El segundo párrafo cuesta 15 € y dura un poco más,"  # cut at the comma

    srt = to_srt(cues[:1])
    assert srt == "1\n00:00:00,000 --> 00:00:02,000\nCorta.\n"
    vtt = to_vtt([build_cues([(3661.25, 3662.5)], ["Hora."])[0]])
    assert vtt == "WEBVTT\n\n01:01:01.250 --> 01:01:02.500\nHora.\n"


def test_wav_is_returned_untouched_and_bad_formats_are_rejected(tmp_path):
    source = tmp_path / "a.wav"
    sf.write(source, tone(1.0), SR)
    assert export_audio(source, "wav", tmp_path / "cache", None) == source
    with pytest.raises(AppError) as exc:
        export_audio(source, "aac", tmp_path / "cache", None)
    assert exc.value.code.value == "VALIDATION_ERROR"
    with pytest.raises(AppError) as exc:
        export_audio(source, "mp3", tmp_path / "cache", None)
    assert exc.value.code.value == "FFMPEG_NOT_FOUND"


@requires_ffmpeg
def test_transcodes_once_and_caches(tmp_path):
    from app.audio.ffmpeg import resolve_binary

    source = tmp_path / "a.wav"
    sf.write(source, tone(2.0), SR, subtype="PCM_24")
    ffmpeg = resolve_binary("ffmpeg")
    for fmt in ("mp3", "ogg", "flac"):
        out = export_audio(source, fmt, tmp_path / "cache", ffmpeg)
        assert out.suffix == f".{fmt}" and out.stat().st_size > 0
        assert export_audio(source, fmt, tmp_path / "cache", ffmpeg) == out  # cached
    flac, sr = sf.read(tmp_path / "cache" / out.name)
    assert sr == SR and flac.size == 2 * SR  # lossless round trip keeps every sample


@requires_ffmpeg
def test_generation_download_in_other_formats(client, engines):  # noqa: F811
    from tests.test_generation import wait_done

    gen_id = client.post("/api/generation", json={"engine": "instructmock", "text": "Hola a todos."}).json()["job_id"]
    wait_done(client, gen_id)
    mp3 = client.get(f"/api/generation/{gen_id}/audio?download=true&format=mp3")
    assert mp3.status_code == 200 and mp3.headers["content-type"] == "audio/mpeg"
    assert mp3.headers["content-disposition"].endswith('.mp3"')
    assert client.get(f"/api/generation/{gen_id}/audio?format=ogg").headers["content-type"] == "audio/ogg"
    assert client.get(f"/api/generation/{gen_id}/audio?format=aac").status_code == 422

    formats = {f["id"]: f for f in client.get("/api/generation/export/formats").json()}
    assert set(formats) == {"wav", "mp3", "ogg", "flac"} and all(f["available"] for f in formats.values())


@requires_ffmpeg
def test_project_formats_and_subtitles(client, engines, settings):  # noqa: F811
    project = _project(client)
    pid = project["id"]
    client.post(f"/api/projects/{pid}/import",
                json={"text": "Hola [pausa:300ms] a todos.\n\n¡Bienvenidos al canal!", "split": "paragraphs"})
    client.post(f"/api/projects/{pid}/generate", json={})
    detail = _wait_project(client, pid)
    assert [s["status"] for s in detail["segments"]] == ["ready", "ready"]

    wav, sr = sf.read(io.BytesIO(client.get(f"/api/projects/{pid}/export").content))
    srt = client.get(f"/api/projects/{pid}/subtitles")
    assert srt.status_code == 200 and srt.headers["content-type"].startswith("application/x-subrip")
    text = srt.text
    assert text.startswith("1\n00:00:00,000 --> ") and "Hola a todos." in text and "[pausa" not in text
    last_end = text.strip().split("\n")[-2].split(" --> ")[1]
    h, m, rest = last_end.split(":")
    assert float(rest.replace(",", ".")) + int(m) * 60 == pytest.approx(wav.shape[0] / sr, abs=0.01)

    vtt = client.get(f"/api/projects/{pid}/subtitles?format=vtt").text
    assert vtt.startswith("WEBVTT") and "¡Bienvenidos al canal!" in vtt

    mp3 = client.get(f"/api/projects/{pid}/export?format=mp3")
    assert mp3.status_code == 200 and mp3.headers["content-type"] == "audio/mpeg"

    names = zipfile.ZipFile(io.BytesIO(client.get(f"/api/projects/{pid}/export?format=zip").content)).namelist()
    assert {n.rsplit(".", 1)[-1] for n in names} >= {"wav", "json", "srt", "vtt"}

    # mastering that changes the timing makes subtitles unreliable: refused with the reason
    client.patch(f"/api/projects/{pid}", json={"settings": {**detail["settings"], "export_postprocess": {
        "trim_silence": {"enabled": True}}}})
    refused = client.get(f"/api/projects/{pid}/subtitles")
    assert refused.status_code == 409 and "subtítulos no coincidirían" in refused.json()["message"]
    assert np.asarray(wav).size > 0
