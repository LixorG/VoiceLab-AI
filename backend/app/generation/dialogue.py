"""Scripts with several speakers: «Ana: Hola» / «Luis: ¿Qué tal?».

The label only says who speaks; it is never read aloud. Each turn is generated with that speaker's voice and the
turns are joined with a pause, so a dialogue sounds like a conversation and not like one person reading a play.

A line without a label continues the previous speaker (a paragraph of the same turn); before the first label it
belongs to the voice selected for the generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# «Ana:», «Dr. Ruiz:», «VOZ EN OFF:» — a short label at the start of a line, followed by what is said. Times
# («12:30»), urls and sentences with a colon in the middle are not labels because the name must open the line.
LABEL = re.compile(r"^[ \t]*(?P<speaker>[^\W\d_][\w .'’-]{0,38})[ \t]*:[ \t]+(?P<text>\S.*)$")
MAX_SPEAKERS = 8
DEFAULT_TURN_PAUSE_MS = 450


@dataclass(frozen=True)
class Turn:
    speaker: str | None  # None = the voice chosen for the generation (lines before the first label)
    text: str


def parse(text: str) -> list[Turn]:
    """Split a script into turns. Consecutive lines of the same speaker become one turn."""
    turns: list[Turn] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = LABEL.match(raw)
        if match:
            turns.append(Turn(match["speaker"].strip(), match["text"].strip()))
        elif turns:
            last = turns[-1]
            turns[-1] = Turn(last.speaker, f"{last.text}\n{line}")
        else:
            turns.append(Turn(None, line))
    return turns


def speakers(text: str) -> list[str]:
    """The labels used in the script, in the order they first appear (that is the order a person reads them)."""
    found: list[str] = []
    for turn in parse(text):
        if turn.speaker and turn.speaker not in found:
            found.append(turn.speaker)
    return found


def looks_like_dialogue(text: str) -> bool:
    """At least two labelled turns: one «Nota: ...» line at the top is not a dialogue."""
    labelled = [t for t in parse(text) if t.speaker]
    return len({t.speaker for t in labelled}) >= 2 or len(labelled) >= 2
