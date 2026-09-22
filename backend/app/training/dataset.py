"""Turn a voice profile's recordings into training clips: utterances of 1.5–15 s with their exact text.

The cuts come from the word timestamps of each reference's full transcript (faster-whisper): an utterance ends at a
sentence end followed by a pause, at any long pause, or at the best pause once it gets long. If the user corrected
the transcript, the corrected words are aligned with the ASR words so the text of every clip is the corrected one.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

import numpy as np

MIN_CLIP_S = 1.5
MAX_CLIP_S = 15.0
LONG_CLIP_S = 10.0  # from here on, any pause is a good place to cut
SENTENCE_GAP_S = 0.2  # pause that, after . ! ? …, ends an utterance
PAUSE_GAP_S = 0.5  # pause that ends an utterance anywhere
SHORT_GAP_S = 0.12  # smallest pause used once the utterance is long
EDGE_PAD_S = (0.12, 0.2)  # silence kept before the first word / after the last one (at most half the gap)
MIN_WORD_PROBABILITY = 0.5  # mean ASR word probability below this = doubtful transcript, clip left out

_SENTENCE_END = re.compile(r"[.!?…]['\"»”)]*$")
_NORM = re.compile(r"[^\w']+", re.UNICODE)


@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class Clip:
    start_s: float
    end_s: float
    text: str
    probability: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def _norm(word: str) -> str:
    return _NORM.sub("", word.lower())


def words_from_timestamps(raw: list[dict]) -> list[Word]:
    return [Word(str(w["word"]).strip(), float(w["start"]), float(w["end"]), float(w.get("probability", 1.0)))
            for w in raw if str(w.get("word", "")).strip()]


def align_edited(words: list[Word], edited: str) -> list[Word]:
    """Give the words of a corrected transcript the timings of the ASR words they replace."""
    tokens = edited.split()
    if [_norm(w.text) for w in words] == [_norm(t) for t in tokens]:
        return [Word(t, w.start, w.end, w.probability) for t, w in zip(tokens, words, strict=True)]
    out: list[Word] = []
    matcher = difflib.SequenceMatcher(a=[_norm(w.text) for w in words], b=[_norm(t) for t in tokens],
                                      autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            out += [Word(tokens[j1 + k], w.start, w.end, w.probability) for k, w in enumerate(words[i1:i2])]
        elif op in ("replace", "insert") and j2 > j1:
            if i2 > i1:  # spread the replaced words over the time of the ASR words they replace
                start, end = words[i1].start, words[i2 - 1].end
            else:  # inserted words: squeeze them next to the previous word
                start = end = out[-1].end if out else (words[i1].start if i1 < len(words) else 0.0)
            step = (end - start) / (j2 - j1)
            # user-typed words are trusted: they get full probability
            out += [Word(tokens[j], start + step * k, start + step * (k + 1)) for k, j in enumerate(range(j1, j2))]
        # "delete": words the user removed are dropped
    return out


def split_utterances(words: list[Word], duration_s: float) -> list[Clip]:
    """Cut a transcribed recording into utterances (see the module docstring for the rules)."""
    clips: list[Clip] = []
    n, start = len(words), 0
    while start < n:
        end = start
        while True:
            word = words[end]
            if end + 1 == n:
                break
            nxt = words[end + 1]
            gap, length = nxt.start - word.end, word.end - words[start].start
            if nxt.end - words[start].start > MAX_CLIP_S:
                # too long with the next word: cut at the longest pause after the minimum length
                candidates = [k for k in range(start, end + 1) if words[k].end - words[start].start >= MIN_CLIP_S]
                if candidates:  # the longest real pause; with none (run-on ASR timings), the longest clip
                    end = max(candidates, key=lambda k: (round(words[k + 1].start - words[k].end, 2),
                                                         words[k].end - words[start].start))
                break
            if length >= MIN_CLIP_S and ((gap >= SENTENCE_GAP_S and _SENTENCE_END.search(word.text))
                                         or gap >= PAUSE_GAP_S or (length >= LONG_CLIP_S and gap >= SHORT_GAP_S)):
                break
            end += 1
        clips.append(_clip(words[start:end + 1], words, start, duration_s))
        start = end + 1
    return clips


def _clip(chunk: list[Word], words: list[Word], first: int, duration_s: float) -> Clip:
    last = first + len(chunk) - 1
    before = chunk[0].start - (words[first - 1].end if first > 0 else 0.0)
    after = (words[last + 1].start if last + 1 < len(words) else duration_s) - chunk[-1].end
    start = max(0.0, chunk[0].start - min(EDGE_PAD_S[0], before / 2))
    end = min(duration_s, chunk[-1].end + min(EDGE_PAD_S[1], after / 2))
    text = " ".join(w.text for w in chunk)
    probability = sum(w.probability for w in chunk) / len(chunk)
    return Clip(round(start, 3), round(end, 3), text, round(probability, 3))


def refine_cuts(clips: list[Clip], audio: np.ndarray, sr: int, window_s: float = 0.12) -> list[Clip]:
    """Where two clips touch (no pause between the words), move the cut to the quietest 10 ms around it so it
    does not fall inside a sound."""
    frame = max(1, sr // 100)
    for left, right in zip(clips, clips[1:], strict=False):
        if right.start_s - left.end_s > 0.02:
            continue
        lo = max(0, int((left.end_s - window_s) * sr))
        hi = min(audio.size, int((left.end_s + window_s) * sr))
        n = (hi - lo) // frame
        if n < 2:
            continue
        rms = np.sqrt(np.mean(audio[lo:lo + n * frame].astype(np.float64).reshape(n, frame) ** 2, axis=1))
        cut = round((lo + (int(np.argmin(rms)) + 0.5) * frame) / sr, 3)
        left.end_s = right.start_s = cut
    return clips


def usable(clip: Clip) -> str | None:
    """None if the clip can be used for training, otherwise why not (Spanish, shown to the user)."""
    if clip.duration_s < MIN_CLIP_S:
        return "demasiado corto"
    if clip.duration_s > MAX_CLIP_S + 0.5:
        return "demasiado largo (sin pausas donde cortar)"
    if clip.probability < MIN_WORD_PROBABILITY:
        return "transcripción dudosa"
    if not any(ch.isalpha() for ch in clip.text):
        return "sin texto"
    return None
