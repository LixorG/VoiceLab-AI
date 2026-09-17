"""Segment boundary helpers based on ASR word timestamps."""

from __future__ import annotations

EDGE_PAD_S = 0.08
MAX_SHIFT_S = 0.6
MIN_SEGMENT_S = 0.5


def word_boundaries(words: list[dict], duration_s: float) -> list[float]:
    """Candidate cut points that never fall inside a word: pauses between words and padded edges."""
    ordered = sorted(words, key=lambda w: w["start"])
    if not ordered:
        return []
    points = [max(0.0, ordered[0]["start"] - EDGE_PAD_S)]
    for a, b in zip(ordered, ordered[1:], strict=False):
        gap = b["start"] - a["end"]
        points.append(a["end"] + gap / 2 if gap > 0 else a["end"])
    points.append(min(duration_s, ordered[-1]["end"] + EDGE_PAD_S))
    return points


def snap_segment(start_s: float, end_s: float, words: list[dict], duration_s: float,
                 max_shift_s: float = MAX_SHIFT_S) -> tuple[float, float]:
    """Move each edge to the nearest word boundary within `max_shift_s`; keep it otherwise."""
    points = word_boundaries(words, duration_s)
    if not points:
        return start_s, end_s

    def nearest(t: float) -> float:
        best = min(points, key=lambda p: abs(p - t))
        return best if abs(best - t) <= max_shift_s + 1e-9 else t

    new_start, new_end = round(nearest(start_s), 2), round(nearest(end_s), 2)
    if new_end - new_start < MIN_SEGMENT_S:
        return start_s, end_s
    return new_start, new_end
