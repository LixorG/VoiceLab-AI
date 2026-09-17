"""SpeechMarkup: inline tags in the text to generate.

Supported (Spanish tags with English aliases, case- and accent-insensitive):
  [pausa:500ms] [pausa:1s] [pausa:1.5s]          pause           ([pause:…])
  [respira] [risa] [suspiro]                      non-verbal sound ([breath] [laugh] [sigh])
  [énfasis]…[/énfasis]                            emphasis span   ([emphasis])
  [susurro]…[/susurro]                            whisper span    ([whisper])
  [tono:grave|agudo|natural]…[/tono]              pitch span      ([pitch:low|high|normal])
  [emoción:feliz] … [/emoción]                    emotion for the following text ([emotion:happy])
A literal bracket is written as \\[ . Engines apply each event only if they really support it (see planner).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

from app.voice_profiles.emotions import EMOTIONS

MAX_PAUSE_MS = 10_000
MAX_TEXT_CHARS = 5000

# Spanish label (normalised) -> emotion id
_EMOTION_ALIASES = {
    **{k: k for k in EMOTIONS},
    **{unicodedata.normalize("NFKD", v).encode("ascii", "ignore").decode().lower(): k
       for k, v in EMOTIONS.items()},
    "enojado": "angry", "alegre": "happy", "tranquilo": "calm", "calmada": "calm", "seria": "serious",
    "misteriosa": "mysterious", "sorprendida": "surprised", "entusiasmada": "excited", "enfadada": "angry",
}
_PITCH_ALIASES = {"grave": "low", "low": "low", "agudo": "high", "high": "high", "natural": "normal",
                  "normal": "normal"}
_SOUND_ALIASES = {"respira": "breath", "respiracion": "breath", "breath": "breath", "risa": "laugh", "laugh": "laugh",
                  "suspiro": "sigh", "sigh": "sigh"}
_SPAN_ALIASES = {"enfasis": "emphasis", "emphasis": "emphasis", "susurro": "whisper", "whisper": "whisper",
                 "tono": "pitch", "pitch": "pitch", "emocion": "emotion", "emotion": "emotion"}

_TAG = re.compile(r"(?<!\\)\[([^\[\]\n]{1,40})\]")


class MarkupError(ValueError):
    def __init__(self, message: str, position: int, tag: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.position = position
        self.tag = tag


@dataclass(frozen=True)
class Style:
    emotion: str | None = None
    emphasis: bool = False
    whisper: bool = False
    pitch: Literal["low", "high", "normal"] | None = None


@dataclass
class TextRun:
    text: str
    style: Style


@dataclass
class Pause:
    ms: int


@dataclass
class Sound:
    kind: Literal["breath", "laugh", "sigh"]
    position: int


@dataclass
class ParsedMarkup:
    events: list[TextRun | Pause | Sound] = field(default_factory=list)
    has_markup: bool = False


def _norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.strip().lower()).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", "", value)


def _parse_pause(arg: str, pos: int, raw: str) -> int:
    match = re.fullmatch(r"(\d+(?:[.,]\d+)?)(ms|s)?", arg)
    if not match:
        raise MarkupError(f"Duración de pausa no válida en «[{raw}]». Usa, por ejemplo, [pausa:500ms] o [pausa:1s].",
                          pos, raw)
    number = float(match.group(1).replace(",", "."))
    ms = int(round(number * (1 if match.group(2) == "ms" else 1000 if match.group(2) == "s" else 1)))
    if match.group(2) is None:  # bare number: treat as milliseconds
        ms = int(round(number))
    if not 0 < ms <= MAX_PAUSE_MS:
        raise MarkupError(f"La pausa debe estar entre 1 ms y {MAX_PAUSE_MS // 1000} s.", pos, raw)
    return ms


def parse(text: str) -> ParsedMarkup:
    result = ParsedMarkup()
    stack: list[tuple[str, int]] = []  # open spans (kind, position)
    style = Style()
    cursor = 0

    def emit_text(chunk: str) -> None:
        chunk = chunk.replace("\\[", "[")
        if chunk:
            if result.events and isinstance(result.events[-1], TextRun) and result.events[-1].style == style:
                result.events[-1].text += chunk
            else:
                result.events.append(TextRun(chunk, style))

    for match in _TAG.finditer(text):
        raw = match.group(1)
        pos = match.start()
        emit_text(text[cursor:pos])
        cursor = match.end()
        result.has_markup = True

        closing = raw.startswith("/")
        name, _, arg = (raw[1:] if closing else raw).partition(":")
        key = _norm(name)
        arg = _norm(arg)

        if closing:
            kind = _SPAN_ALIASES.get(key)
            if kind is None:
                raise MarkupError(f"Etiqueta de cierre desconocida «[{raw}]».", pos, raw)
            if kind == "emotion":
                style = Style(None, style.emphasis, style.whisper, style.pitch)
                continue
            if not stack or stack[-1][0] != kind:
                raise MarkupError(f"«[{raw}]» no tiene una etiqueta de apertura correspondiente.", pos, raw)
            stack.pop()
            style = Style(style.emotion, emphasis=style.emphasis and kind != "emphasis",
                          whisper=style.whisper and kind != "whisper", pitch=None if kind == "pitch" else style.pitch)
            continue

        if key in ("pausa", "pause"):
            if not arg:
                raise MarkupError("Indica la duración de la pausa, por ejemplo [pausa:500ms].", pos, raw)
            result.events.append(Pause(_parse_pause(arg, pos, raw)))
        elif key in _SOUND_ALIASES and not arg:
            result.events.append(Sound(_SOUND_ALIASES[key], pos))
        elif _SPAN_ALIASES.get(key) == "emotion":
            emotion = _EMOTION_ALIASES.get(arg)
            if emotion is None:
                raise MarkupError(f"Emoción desconocida «{arg or '(vacía)'}». Opciones: "
                                  + ", ".join(v.lower() for v in EMOTIONS.values()) + ".", pos, raw)
            style = Style(emotion, style.emphasis, style.whisper, style.pitch)
        elif _SPAN_ALIASES.get(key) in ("emphasis", "whisper") and not arg:
            kind = _SPAN_ALIASES[key]
            stack.append((kind, pos))
            style = Style(style.emotion, emphasis=style.emphasis or kind == "emphasis",
                          whisper=style.whisper or kind == "whisper", pitch=style.pitch)
        elif _SPAN_ALIASES.get(key) == "pitch":
            pitch = _PITCH_ALIASES.get(arg)
            if pitch is None:
                raise MarkupError("Tono no válido: usa [tono:grave], [tono:agudo] o [tono:natural].", pos, raw)
            stack.append(("pitch", pos))
            style = Style(style.emotion, style.emphasis, style.whisper, pitch)  # type: ignore[arg-type]
        else:
            raise MarkupError(f"Etiqueta desconocida «[{raw}]». Escribe \\[ para un corchete literal.", pos, raw)

    emit_text(text[cursor:])
    if stack:
        kind, pos = stack[-1]
        names = {"emphasis": "énfasis", "whisper": "susurro", "pitch": "tono"}
        raise MarkupError(f"Falta cerrar la etiqueta [{names[kind]}] con [/{names[kind]}].", pos, kind)
    return result
