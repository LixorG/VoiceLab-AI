"""Training clips from transcribed recordings: cuts at pauses, corrected transcripts, cut refinement, filters."""

from __future__ import annotations

import numpy as np

from app.training.dataset import (
    MAX_CLIP_S,
    Clip,
    Word,
    align_edited,
    refine_cuts,
    split_utterances,
    usable,
    words_from_timestamps,
)


def words(spec: list[tuple[str, float, float]], probability: float = 0.95) -> list[Word]:
    return [Word(t, s, e, probability) for t, s, e in spec]


def test_words_from_timestamps_skips_empty_words():
    raw = [{"word": " Hola", "start": 0.1, "end": 0.4, "probability": 0.9}, {"word": " ", "start": 0.4, "end": 0.5}]
    assert words_from_timestamps(raw) == [Word("Hola", 0.1, 0.4, 0.9)]


def test_utterances_end_at_sentence_pauses_and_long_pauses():
    ws = words([("Hola,", 0.0, 0.5), ("¿qué", 0.6, 0.9), ("tal", 0.95, 1.3), ("estás?", 1.35, 1.9),  # . + 0.3 s
                ("Muy", 2.2, 2.5), ("bien", 2.55, 2.9), ("gracias", 3.0, 3.8),  # no punctuation, 0.6 s pause
                ("y", 4.4, 4.5), ("tú", 4.55, 4.9), ("también", 4.95, 6.0)])
    clips = split_utterances(ws, 6.7)
    assert [c.text for c in clips] == ["Hola, ¿qué tal estás?", "Muy bien gracias", "y tú también"]
    # a margin is kept around the words: at most half the pause, 0.12 s before and 0.2 s after
    assert clips[0].start_s == 0.0 and clips[0].end_s == 2.05
    assert clips[1].start_s == 2.08 and clips[1].end_s == 4.0
    assert clips[2].end_s == 6.2


def test_a_short_sentence_is_not_cut_before_the_minimum_length():
    ws = words([("Sí.", 0.0, 0.4), ("Claro", 0.8, 1.2), ("que", 1.25, 1.4), ("sí.", 1.45, 2.0)])
    assert [c.text for c in split_utterances(ws, 2.5)] == ["Sí. Claro que sí."]


def test_long_run_on_speech_is_cut_at_the_longest_pause_before_the_limit():
    # 40 words of 0.4 s, touching (no pauses) except one 0.1 s gap at 6 s
    spec, t = [], 0.0
    for i in range(40):
        spec.append((f"w{i}", t, t + 0.4))
        t += 0.4 + (0.1 if i == 14 else 0.0)
    clips = split_utterances(words(spec), t)
    assert clips[0].text.split()[-1] == "w14"  # the only pause wins over a longer clip
    assert all(c.duration_s <= MAX_CLIP_S + 0.3 for c in clips)
    assert " ".join(c.text for c in clips).split() == [f"w{i}" for i in range(40)]  # nothing lost or repeated


def test_without_any_pause_the_clip_is_as_long_as_allowed():
    spec = [(f"w{i}", i * 0.5, i * 0.5 + 0.5) for i in range(40)]
    clips = split_utterances(words(spec), 20.0)
    assert 14.0 <= clips[0].duration_s <= MAX_CLIP_S


def test_corrected_transcript_keeps_timings():
    asr = words([("I", 0.0, 0.2), ("was", 0.25, 0.5), ("sneak", 0.6, 0.9), ("out", 0.95, 1.2), ("a", 1.25, 1.3),
                 ("nutty's", 1.35, 1.9), ("today.", 2.0, 2.5)])
    fixed = align_edited(asr, "I was sneaking out of Nate's today.")
    assert [w.text for w in fixed] == ["I", "was", "sneaking", "out", "of", "Nate's", "today."]
    assert fixed[0].start == 0.0 and fixed[-1].end == 2.5 and fixed[1].start == 0.25
    assert fixed[2].start == 0.6 and fixed[5].end == 1.9  # replaced words share the time of the ASR words
    # same words, only capitalisation/punctuation changed: timings copied one to one
    same = align_edited(asr[:2], "i WAS")
    assert [(w.text, w.start) for w in same] == [("i", 0.0), ("WAS", 0.25)]
    # inserted and deleted words
    inserted = align_edited(asr[:2], "I really was")
    assert [w.text for w in inserted] == ["I", "really", "was"] and inserted[1].start == 0.2
    assert [w.text for w in align_edited(asr[:3], "I sneak")] == ["I", "sneak"]


def test_touching_clips_are_cut_at_the_quietest_point():
    sr = 1000
    audio = np.full(3 * sr, 0.5, dtype=np.float32)
    audio[1450:1470] = 0.0  # a short dip 50 ms after the ASR boundary
    clips = [Clip(0.0, 1.4, "a", 1.0), Clip(1.4, 3.0, "b", 1.0), Clip(3.2, 3.5, "c", 1.0)]
    refine_cuts(clips, audio, sr)
    assert clips[0].end_s == clips[1].start_s and 1.45 <= clips[0].end_s <= 1.47
    assert clips[2].start_s == 3.2  # a real gap is left alone


def test_usable_filters():
    assert usable(Clip(0, 5, "Hola a todos", 0.9)) is None
    assert usable(Clip(0, 1, "Hola", 0.9)) == "demasiado corto"
    assert usable(Clip(0, 16, "Hola", 0.9)).startswith("demasiado largo")
    assert usable(Clip(0, 5, "Hola a todos", 0.3)) == "transcripción dudosa"
    assert usable(Clip(0, 5, "...", 0.9)) == "sin texto"
