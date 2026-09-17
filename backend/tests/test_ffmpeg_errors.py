"""FFmpeg/ffprobe failure paths with a simulated subprocess (no real binaries involved)."""

from __future__ import annotations

import json
import subprocess
import types

import pytest

from app.audio import ffmpeg as ffmpeg_module
from app.audio.ffmpeg import decode_to_wav, probe, resolve_binary
from app.core.errors import AppError, ErrorCode


def _proc(returncode=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_missing_binary(monkeypatch):
    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda name: None)
    with pytest.raises(AppError) as err:
        resolve_binary("ffmpeg")
    assert err.value.code is ErrorCode.FFMPEG_NOT_FOUND
    asked: list[str] = []
    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda name: asked.append(name) or f"{name}.exe")
    assert resolve_binary("ffprobe", "C:/custom/ffprobe") == "C:/custom/ffprobe.exe"
    assert resolve_binary("ffprobe") == "ffprobe.exe" and asked == ["C:/custom/ffprobe", "ffprobe"]


@pytest.mark.parametrize(
    ("outcome", "code"),
    [
        (subprocess.TimeoutExpired("ffprobe", 30), ErrorCode.AUDIO_DECODE_ERROR),
        (_proc(returncode=1, stderr="Invalid data found"), ErrorCode.AUDIO_DECODE_ERROR),
        (_proc(stdout="not json"), ErrorCode.AUDIO_DECODE_ERROR),
        (_proc(stdout=json.dumps({"streams": [{"codec_type": "video"}], "format": {}})), ErrorCode.UNSUPPORTED_FORMAT),
        (_proc(stdout=json.dumps({"streams": [{"codec_type": "audio", "sample_rate": "x"}], "format": {}})),
         ErrorCode.AUDIO_DECODE_ERROR),
    ],
)
def test_probe_failures(monkeypatch, tmp_path, outcome, code):
    def run(*args, **kwargs):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(ffmpeg_module.subprocess, "run", run)
    with pytest.raises(AppError) as err:
        probe(tmp_path / "a.wav", "ffprobe")
    assert err.value.code is code


def test_probe_reads_format_duration_and_bitrate(monkeypatch, tmp_path):
    data = {"streams": [{"codec_type": "audio", "codec_name": "mp3", "sample_rate": "44100", "channels": 2}],
            "format": {"format_name": "mp3", "duration": "3.5", "bit_rate": "128000"}}
    monkeypatch.setattr(ffmpeg_module.subprocess, "run", lambda *a, **k: _proc(stdout=json.dumps(data)))
    info = probe(tmp_path / "a.mp3", "ffprobe")
    assert (info.codec, info.sample_rate, info.channels, info.duration_s, info.bit_rate) == ("mp3", 44100, 2, 3.5,
                                                                                            128000)


def test_decode_retries_without_soxr_and_reports_failures(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    dst = tmp_path / "out.wav"

    def run(cmd, **kwargs):
        calls.append(cmd)
        if "aresample=resampler=soxr" in cmd:
            return _proc(returncode=1, stderr="No such filter option: soxr")
        (tmp_path / "out.tmp.wav").write_bytes(b"RIFF")
        return _proc()

    monkeypatch.setattr(ffmpeg_module.subprocess, "run", run)
    decode_to_wav(tmp_path / "in.mp3", dst, "ffmpeg", 24_000)
    assert len(calls) == 2 and "aresample=resampler=soxr" not in calls[1] and dst.exists()

    monkeypatch.setattr(ffmpeg_module.subprocess, "run", lambda cmd, **k: _proc(returncode=1, stderr="corrupt"))
    with pytest.raises(AppError) as err:
        decode_to_wav(tmp_path / "in.mp3", tmp_path / "bad.wav", "ffmpeg", 24_000)
    assert err.value.code is ErrorCode.AUDIO_DECODE_ERROR and not (tmp_path / "bad.tmp.wav").exists()

    def timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(ffmpeg_module.subprocess, "run", timeout)
    with pytest.raises(AppError):
        decode_to_wav(tmp_path / "in.mp3", tmp_path / "slow.wav", "ffmpeg", 24_000)
