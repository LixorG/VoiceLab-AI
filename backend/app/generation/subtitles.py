"""SRT / WebVTT subtitles for an assembled project, timed from the real audio of each segment.

Each segment's span comes from `assembly.timeline()`, so the cues line up with the exported WAV. The text shown is
what the user wrote (markup tags removed, numbers not normalised: subtitles are read, not spoken). A long segment is
split at sentence boundaries into several cues, sharing its span in proportion to their length — an estimate inside
the segment (the models give no word timings), exact at segment edges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.generation.markup import MarkupError, TextRun, parse

MAX_CUE_CHARS = 84  # two lines of ~42 characters, the usual subtitle limit
_SENTENCE = re.compile(r"(?<=[.!?…;:])\s+")
_CLAUSE = re.compile(r"(?<=,)\s+")


@dataclass(frozen=True)
class Cue:
    start_s: float
    end_s: float
    text: str


def spoken_text(text: str, markup: bool = True) -> str:
    """The segment text without SpeechMarkup tags ([pausa:500ms], [emoción:feliz]…)."""
    if not markup:
        return " ".join(text.split())
    try:
        events = parse(text).events
    except MarkupError:
        return " ".join(text.split())
    return " ".join(" ".join(e.text for e in events if isinstance(e, TextRun)).split())


def _pack(pieces: list[str]) -> list[str]:
    """Greedily join consecutive pieces while they fit in one cue."""
    chunks: list[str] = []
    for piece in pieces:
        if chunks and len(chunks[-1]) + 1 + len(piece) <= MAX_CUE_CHARS:
            chunks[-1] = f"{chunks[-1]} {piece}"
        else:
            chunks.append(piece)
    return chunks


def _chunks(text: str) -> list[str]:
    if len(text) <= MAX_CUE_CHARS:
        return [text]
    # sentences first; a sentence that is still too long is cut at its commas before falling back to words
    chunks: list[str] = []
    for sentence in _pack(_SENTENCE.split(text)):
        chunks.extend(_pack(_CLAUSE.split(sentence)) if len(sentence) > MAX_CUE_CHARS else [sentence])
    # a clause still longer than the limit is split at word boundaries
    out: list[str] = []
    for chunk in chunks:
        words, line = chunk.split(), ""
        for word in words:
            if line and len(line) + 1 + len(word) > MAX_CUE_CHARS:
                out.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        if line:
            out.append(line)
    return out


def build_cues(spans: list[tuple[float, float]], texts: list[str]) -> list[Cue]:
    cues: list[Cue] = []
    for (start, end), text in zip(spans, texts, strict=True):
        if not text:
            continue
        chunks = _chunks(text)
        total = sum(len(c) for c in chunks) or 1
        cursor = start
        for index, chunk in enumerate(chunks):
            stop = end if index == len(chunks) - 1 else cursor + (end - start) * len(chunk) / total
            cues.append(Cue(round(cursor, 3), round(stop, 3), _wrap(chunk)))
            cursor = stop
    return cues


def _wrap(text: str, width: int = MAX_CUE_CHARS // 2) -> str:
    """At most two balanced lines."""
    if len(text) <= width:
        return text
    words = text.split()
    best, best_gap = text, len(text)
    for i in range(1, len(words)):
        first, second = " ".join(words[:i]), " ".join(words[i:])
        gap = abs(len(first) - len(second))
        if gap < best_gap:
            best, best_gap = f"{first}\n{second}", gap
    return best


def _timestamp(seconds: float, separator: str) -> str:
    ms = int(round(seconds * 1000))
    h, rest = divmod(ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{separator}{ms:03d}"


def to_srt(cues: list[Cue]) -> str:
    blocks = [f"{i}\n{_timestamp(c.start_s, ',')} --> {_timestamp(c.end_s, ',')}\n{c.text}"
              for i, c in enumerate(cues, start=1)]
    return "\n\n".join(blocks) + "\n"


def to_vtt(cues: list[Cue]) -> str:
    blocks = [f"{_timestamp(c.start_s, '.')} --> {_timestamp(c.end_s, '.')}\n{c.text}" for c in cues]
    return "WEBVTT\n\n" + "\n\n".join(blocks) + "\n"
