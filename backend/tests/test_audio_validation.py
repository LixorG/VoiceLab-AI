import pytest

from app.audio.ffmpeg import ProbeResult
from app.audio.storage import AudioStorage
from app.audio.validation import extension_of, sanitize_filename, validate_probe
from app.core.errors import AppError, ErrorCode


@pytest.mark.parametrize(("raw", "expected"), [
    ("../../etc/passwd.wav", "passwd.wav"),
    ("C:\\Users\\x\\..\\voz final?.mp3", "voz final_.mp3"),
    ("grabación_ñ.flac", "grabación_ñ.flac"),
    ("....", "audio"),
    ("", "audio"),
])
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected


def test_sanitize_truncates_keeping_extension():
    name = sanitize_filename("a" * 300 + ".wav")
    assert len(name) == 120 and name.endswith(".wav")


@pytest.mark.parametrize("name", ["virus.exe", "voz.wav.exe", "sin_extension", "clip.mp4"])
def test_extension_whitelist_rejects(name):
    with pytest.raises(AppError) as exc:
        extension_of(name)
    assert exc.value.code is ErrorCode.UNSUPPORTED_FORMAT


@pytest.mark.parametrize("name", ["a.WAV", "b.mp3", "c.flac", "d.m4a", "e.ogg", "f.opus", "g.aiff", "h.aif"])
def test_extension_whitelist_accepts(name):
    assert extension_of(name) == name.rsplit(".", 1)[1].lower()


def _probe(fmt="wav", duration=5.0):
    return ProbeResult(format_name=fmt, codec="pcm_s16le", sample_rate=44100, channels=2, duration_s=duration)


def test_probe_must_match_extension():
    with pytest.raises(AppError) as exc:
        validate_probe("wav", _probe(fmt="mp3"), 600)
    assert exc.value.code is ErrorCode.UNSUPPORTED_FORMAT
    validate_probe("m4a", _probe(fmt="mov,mp4,m4a,3gp,3g2,mj2"), 600)


def test_probe_duration_limits():
    with pytest.raises(AppError) as exc:
        validate_probe("wav", _probe(duration=700), 600)
    assert exc.value.code is ErrorCode.AUDIO_TOO_LONG
    with pytest.raises(AppError):
        validate_probe("wav", _probe(duration=0.05), 600)


def test_storage_rejects_non_hash_paths(tmp_path):
    storage = AudioStorage(tmp_path)
    for bad in ("../../evil", "a" * 63, "Z" * 64):
        with pytest.raises(ValueError):
            storage.processed_path(bad)
    with pytest.raises(ValueError):
        storage.original_path("a" * 64, "exe")
    assert storage.processed_path("a" * 64).parent == storage.processed_dir
