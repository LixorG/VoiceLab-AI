import numpy as np
import pytest

from app.audio.analysis import SEGMENT_MAX_S, SEGMENT_MIN_S, analyze
from app.audio.processing import PEAK_CEILING_DBFS, apply_gain, normalization_gain_db, waveform_peaks
from tests.audio_fixtures import speechlike

SR = 24_000


def test_clean_speechlike_signal_scores_high():
    a = analyze(speechlike(words=20), SR)
    assert a.speech_detected
    assert a.snr_db is not None and a.snr_db > 30
    assert a.clipped_samples == 0
    assert 0.4 < a.speech_ratio < 0.95
    assert len(a.speech_regions) >= 5  # pauses < 300 ms are merged
    assert a.quality_score >= 80
    assert a.suggested_segments
    for seg in a.suggested_segments:
        assert SEGMENT_MIN_S - 1e-6 <= seg.end_s - seg.start_s <= SEGMENT_MAX_S + 1e-6


def test_noise_lowers_snr_and_score():
    clean = analyze(speechlike(words=20), SR)
    noisy = analyze(speechlike(words=20, noise_db=-28), SR)
    assert noisy.snr_db < clean.snr_db
    assert noisy.quality_score < clean.quality_score
    assert any("Ruido" in w for w in noisy.warnings)


def test_silence_has_no_speech():
    a = analyze(np.zeros(SR * 5, dtype=np.float32), SR)
    assert not a.speech_detected
    assert a.quality_score <= 20
    assert any("No se detectó voz" in w for w in a.warnings)


def test_clipping_detected():
    a = analyze(np.clip(speechlike(words=20, gain=3.0), -1, 1), SR)
    assert a.clipping_ratio > 0.001
    assert any("saturación" in w for w in a.warnings)
    clip = next(c for c in a.quality_components if c.id == "clipping")
    assert clip.score < 0.5


def test_short_reference_warns():
    a = analyze(speechlike(words=2), SR)
    assert a.effective_speech_s < 3
    assert any("3 s" in w for w in a.warnings)


def test_low_source_sample_rate_penalised():
    samples = speechlike(words=20)
    assert analyze(samples, SR, source_sample_rate=8000).quality_score < analyze(samples, SR).quality_score


@pytest.mark.parametrize("peak_dbfs", [-30.0, -3.0])
def test_normalization_respects_peak_ceiling(peak_dbfs):
    gain = normalization_gain_db(loudness_lufs=-35.0, rms_dbfs=-38.0, peak_dbfs=peak_dbfs)
    assert peak_dbfs + gain <= PEAK_CEILING_DBFS + 1e-9


def test_apply_gain_and_peaks():
    x = np.array([0.1, -0.5, 0.25, 0.0], dtype=np.float32)
    assert np.allclose(apply_gain(x, 6.0206), x * 2, atol=1e-4)
    peaks = waveform_peaks(np.tile(x, 100), 50)
    assert len(peaks) == 50 and max(peaks) == pytest.approx(0.5)
