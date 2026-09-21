"""Post-processing: DSP chain (numpy + FFmpeg) and mastering API over generations — simulated engines, no GPU."""

from __future__ import annotations

import io
import shutil

import numpy as np
import pytest
import soundfile as sf

from app.postprocess.config import PostProcessConfig
from app.postprocess.processor import (
    apply_fades,
    capabilities,
    loudness_lufs,
    noise_floor_dbfs,
    process,
    trim_silence,
)
from tests.audio_fixtures import requires_ffmpeg
from tests.test_advanced_generation import engines as engines  # noqa: F401  (fixture re-export)
from tests.test_generation import wait_done

SR = 24_000


def tone(freq: float = 220.0, seconds: float = 2.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_hz(audio: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(audio.size)))
    return float(np.fft.rfftfreq(audio.size, 1 / SR)[np.argmax(spectrum)])


def cfg(**kw) -> PostProcessConfig:
    return PostProcessConfig.model_validate(kw)


# ---------------------------------------------------------------- pure DSP
def test_default_config_is_inactive_and_changes_nothing():
    audio = tone()
    out, report = process(audio, SR, PostProcessConfig(), ffmpeg=None)
    assert not PostProcessConfig().is_active and report.steps == [] and np.array_equal(out, audio)


def test_trim_silence_keeps_padding():
    audio = np.concatenate([np.zeros(SR), tone(seconds=1.0), np.zeros(SR // 2)]).astype(np.float32)
    out, cut_start, cut_end = trim_silence(audio, SR, threshold_db=-45, padding_ms=100)
    assert abs(cut_start - 0.9) < 0.02 and abs(cut_end - 0.4) < 0.02
    assert abs(out.size / SR - 1.2) < 0.03
    assert trim_silence(np.zeros(SR, dtype=np.float32), SR, -45, 100)[0].size == SR  # all silence: untouched


def test_fades_start_and_end_at_zero():
    out = apply_fades(np.ones(SR, dtype=np.float32), SR, 50, 100)
    assert out[0] == 0 and abs(out[-1]) < 1e-3 and out[SR // 2] == 1


def test_loudness_and_peak_normalization():
    audio = tone(amp=0.05)
    out, report = process(audio, SR, cfg(loudness={"enabled": True, "target_lufs": -20}), ffmpeg=None)
    assert abs(loudness_lufs(out, SR) + 20) < 0.3 and report.after.loudness_lufs == pytest.approx(-20, abs=0.3)

    out, report = process(audio, SR, cfg(peak={"enabled": True, "target_dbfs": -3}), ffmpeg=None)
    assert 20 * np.log10(np.max(np.abs(out))) == pytest.approx(-3, abs=0.05)

    # too loud for the ceiling: the peak wins and the report says so
    out, report = process(audio, SR, cfg(loudness={"enabled": True, "target_lufs": -10},
                                         peak={"enabled": True, "target_dbfs": -12}), ffmpeg=None)
    assert 20 * np.log10(np.max(np.abs(out))) == pytest.approx(-12, abs=0.05)
    assert any("por debajo del objetivo" in w for w in report.warnings)
    assert [s.id for s in report.steps] == ["loudness", "peak"]


def test_short_audio_skips_loudness_with_warning():
    _, report = process(tone(seconds=0.2), SR, cfg(loudness={"enabled": True}), ffmpeg=None)
    assert report.steps == [] and "demasiado corto" in report.warnings[0]


def test_ffmpeg_processors_unavailable_are_reported():
    caps = capabilities(None)
    assert not caps.processors["pitch_shift"] and "FFmpeg" in caps.reasons["pitch_shift"]
    out, report = process(tone(), SR, cfg(pitch_shift={"enabled": True, "semitones": 3}), ffmpeg=None)
    assert report.steps == [] and "No se aplicó" in report.warnings[0]


@requires_ffmpeg
def test_pitch_shift_and_time_stretch_with_rubberband():
    ffmpeg = shutil.which("ffmpeg")
    if not capabilities(ffmpeg).processors["pitch_shift"]:
        pytest.skip("FFmpeg sin rubberband")
    audio = tone(220.0)
    out, report = process(audio, SR, cfg(pitch_shift={"enabled": True, "semitones": 12}), ffmpeg)
    assert dominant_hz(out) == pytest.approx(440.0, rel=0.03) and abs(out.size - audio.size) < SR * 0.05

    out, report = process(audio, SR, cfg(time_stretch={"enabled": True, "rate": 1.5}), ffmpeg, "speed")
    assert out.size / SR == pytest.approx(2.0 / 1.5, rel=0.03) and dominant_hz(out) == pytest.approx(220, rel=0.03)
    assert any("«speed»" in w for w in report.warnings)


@requires_ffmpeg
def test_denoise_reduces_noise_floor():
    ffmpeg = shutil.which("ffmpeg")
    rng = np.random.default_rng(0)
    noisy = (tone() + 0.02 * rng.standard_normal(2 * SR)).astype(np.float32)
    noisy[: SR // 2] = 0.02 * rng.standard_normal(SR // 2)  # noise-only lead-in
    out, report = process(noisy, SR, cfg(denoise={"enabled": True, "strength": "strong"}), ffmpeg)
    assert report.steps[0].id == "denoise" and np.std(out[SR // 4: SR // 2]) < np.std(noisy[SR // 4: SR // 2])


def test_noise_floor_measurement():
    rng = np.random.default_rng(1)
    quiet = (0.001 * rng.standard_normal(SR)).astype(np.float32)  # ~-60 dBFS background
    assert noise_floor_dbfs(quiet, SR) == pytest.approx(-60, abs=2)
    assert noise_floor_dbfs(np.zeros(SR // 20, np.float32), SR) is None  # too short to tell


def test_clean_audio_is_left_untouched_by_denoise():
    """A clean take has nothing to remove: filtering it anyway is what made voices sound metallic."""
    audio = tone()
    audio[: SR // 2] = 0.0  # digital silence around the voice, like a TTS output
    out, report = process(audio, SR, cfg(denoise={"enabled": True, "strength": "strong"}), ffmpeg="ffmpeg")
    assert np.array_equal(out, audio)
    assert report.steps[0].id == "denoise" and report.steps[0].detail.startswith("Sin cambios: el audio ya está limpio")
    assert not report.warnings


@requires_ffmpeg
def test_denoise_uses_the_measured_floor_without_tracking(monkeypatch):
    from app.postprocess import processor

    seen = []
    real = processor._run_ffmpeg
    monkeypatch.setattr(processor, "_run_ffmpeg", lambda a, sr, f, ff: seen.append(f) or real(a, sr, f, ff))
    rng = np.random.default_rng(0)
    noisy = (tone() + 0.02 * rng.standard_normal(2 * SR)).astype(np.float32)
    _, report = process(noisy, SR, cfg(denoise={"enabled": True, "strength": "medium"}), shutil.which("ffmpeg"))
    assert seen and seen[0].startswith("afftdn=nr=12:nf=-") and seen[0].endswith(":tn=0")
    assert "ruido de fondo medido" in report.steps[0].detail


@requires_ffmpeg
def test_speed_uses_atempo_and_pitch_warnings_follow_the_measurements(monkeypatch):
    from app.postprocess import processor

    ffmpeg = shutil.which("ffmpeg")
    seen = []
    real = processor._run_ffmpeg
    monkeypatch.setattr(processor, "_run_ffmpeg", lambda a, sr, f, ff: seen.append(f) or real(a, sr, f, ff))
    out, report = process(tone(), SR, cfg(time_stretch={"enabled": True, "rate": 0.95}), ffmpeg)
    assert seen[-1] == "atempo=0.9500" and "atempo" in report.steps[0].detail and not report.warnings
    _, report = process(tone(), SR, cfg(time_stretch={"enabled": True, "rate": 1.2}), ffmpeg)
    assert any("10 %" in w for w in report.warnings)

    if not capabilities(ffmpeg).processors["pitch_shift"]:
        pytest.skip("FFmpeg sin rubberband")
    for semitones, expected in ((0.5, None), (1.5, "ya se nota"), (3, "robótica")):
        _, report = process(tone(), SR, cfg(pitch_shift={"enabled": True, "semitones": semitones}), ffmpeg)
        assert (expected is None and not report.warnings) or any(expected in w for w in report.warnings)


# ---------------------------------------------------------------- API
@requires_ffmpeg
def test_generation_with_postprocess_keeps_raw_output(client, engines):  # noqa: F811
    body = {"engine": "instructmock", "text": "Hola a todos", "params": {"seed": 3},
            "postprocess": {"peak": {"enabled": True, "target_dbfs": -6}, "fades": {"enabled": True}}}
    done = wait_done(client, client.post("/api/generation", json=body).json()["job_id"])
    assert done["status"] == "COMPLETED"
    assert [s["id"] for s in done["postprocess"]["report"]["steps"]] == ["peak", "fades"]

    final, _ = sf.read(io.BytesIO(client.get(done["audio_url"]).content))
    raw, _ = sf.read(io.BytesIO(client.get(done["raw_audio_url"]).content))
    assert 20 * np.log10(np.max(np.abs(final))) == pytest.approx(-6, abs=0.1)
    assert not np.allclose(np.max(np.abs(raw)), np.max(np.abs(final)), atol=1e-3)
    assert sf.info(io.BytesIO(client.get(done["raw_audio_url"]).content)).subtype == "FLOAT"


@requires_ffmpeg
def test_remaster_and_clear(client, engines):  # noqa: F811
    done = wait_done(client, client.post("/api/generation", json={"engine": "instructmock",
                                                                  "text": "Hola"}).json()["job_id"])
    assert done["postprocess"] is None
    res = client.post(f"/api/generation/{done['id']}/postprocess", json={"peak": {"enabled": True, "target_dbfs": -12}})
    assert res.status_code == 200
    mastered = res.json()
    assert mastered["postprocess"]["config"]["peak"]["target_dbfs"] == -12
    assert mastered["audio_url"] != done["audio_url"] or mastered["updated_at"] != done["updated_at"]
    final, _ = sf.read(io.BytesIO(client.get(mastered["audio_url"]).content))
    assert 20 * np.log10(np.max(np.abs(final))) == pytest.approx(-12, abs=0.1)

    cleared = client.delete(f"/api/generation/{done['id']}/postprocess").json()
    assert cleared["postprocess"] is None
    again, _ = sf.read(io.BytesIO(client.get(cleared["audio_url"]).content))
    raw, _ = sf.read(io.BytesIO(client.get(cleared["raw_audio_url"]).content))
    assert np.allclose(again, raw, atol=1e-4)  # PCM 24-bit round trip of the untouched output

    bad = client.post(f"/api/generation/{done['id']}/postprocess", json={"pitch_shift": {"semitones": 40}})
    assert bad.status_code == 422
    assert client.post("/api/generation/nope/postprocess", json={}).status_code == 404


@requires_ffmpeg
def test_crossfade_reassembles_saved_segments(client, engines):  # noqa: F811
    body = {"engine": "instructmock", "text": "Hola. [emoción:feliz]Adiós[/emoción]", "params": {"seed": 1}}
    done = wait_done(client, client.post("/api/generation", json=body).json()["job_id"])
    assert len(done["segments"]) == 2
    seg_total = sum(s["duration_s"] for s in done["segments"])
    assert done["duration_s"] == pytest.approx(seg_total - 0.04, abs=0.002)

    res = client.post(f"/api/generation/{done['id']}/postprocess", json={"crossfade_ms": 200}).json()
    assert res["duration_s"] == pytest.approx(seg_total - 0.2, abs=0.002)
    assert res["postprocess"]["report"]["steps"][0]["id"] == "crossfade"

    simple = wait_done(client, client.post("/api/generation", json={"engine": "instructmock",
                                                                    "text": "Hola"}).json()["job_id"])
    res = client.post(f"/api/generation/{simple['id']}/postprocess", json={"crossfade_ms": 100}).json()
    assert "varios segmentos" in res["postprocess"]["report"]["warnings"][0]


def test_capabilities_endpoint(client):
    caps = client.get("/api/generation/postprocess/capabilities").json()
    assert caps["processors"]["loudness"] is True and "pitch_shift" in caps["processors"]
