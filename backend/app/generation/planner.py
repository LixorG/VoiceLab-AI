"""Turn parsed markup + expression settings into engine-aware segments.

Each style attribute is kept only if the selected engine can really apply it (capabilities):
- instruction engines (Qwen CustomVoice / VoiceDesign): emotion, intensity, whisper, emphasis and pitch become an
  English style instruction appended to the user's own instruction;
- segmentation engines (F5, E2, Qwen clone): emotion is applied by switching to a reference tagged with that emotion
  in the voice profile (intensity cannot be expressed);
- anything else is dropped with a Spanish warning. Adjacent text with the same *effective* style is merged, so
  unsupported tags never split sentences (which would hurt prosody for no benefit).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.engines.base import ControlSource, EngineCapabilities, GenericControl
from app.generation.markup import ParsedMarkup, Pause, Sound, Style, TextRun
from app.voice_profiles.emotions import EMOTIONS

BREATH_PAUSE_MS = 300
# Natural pauses when an engine generates sentence by sentence (engines with `sentence_chunks`).
SENTENCE_PAUSE_MS = 350
PARAGRAPH_PAUSE_MS = 750
CLAUSE_PAUSE_MS = 150  # a sentence too long for one call is cut at a comma
MIN_SENTENCE_CHARS = 40  # shorter sentences ("First." / "Yes!") are generated together with a neighbour
SHOUT_MIN_WORDS = 3  # shorter all-caps runs are usually acronyms (IA, NASA, ONU)

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def soften_shouting(text: str) -> tuple[str, bool]:
    """Lower-case runs of >= 3 consecutive ALL-CAPS words, keeping sentence capitals and short acronyms.

    TTS models do not read capitals as emphasis; character-based ones (F5/E2) even mispronounce them.
    """
    words = list(_WORD.finditer(text))
    shouting = [w.group(0).isupper() and len(w.group(0)) > 0 for w in words]
    chars = list(text)
    changed = False
    i = 0
    while i < len(words):
        if not shouting[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(words) and shouting[j + 1]:
            j += 1
        if j - i + 1 >= SHOUT_MIN_WORDS:
            for w in words[i:j + 1]:
                lowered = w.group(0).lower()
                # Keep a capital where a sentence starts (text start, or after . ! ? and the Spanish ¡ ¿).
                before = text[:w.start()].rstrip()
                if not before or before[-1] in ".!?¡¿":
                    lowered = lowered[0].upper() + lowered[1:]
                chars[w.start():w.end()] = lowered
            changed = True
        i = j + 1
    return ("".join(chars), True) if changed else (text, False)


_SENTENCE_END = re.compile(r"(?<=[.!?…;])\s+")
_CLAUSE_END = re.compile(r"(?<=[,:])\s+")


_PARAGRAPH = re.compile(r"\n\s*\n")
_ENDS_PARAGRAPH = re.compile(r"\n\s*\n\s*$")
_STARTS_PARAGRAPH = re.compile(r"\s*\n\s*\n")


def sentence_chunks(text: str, max_chars: int) -> list[tuple[str, int]]:
    """(chunk, pause after in ms): one chunk per sentence, paragraphs kept; the last pause is 0.

    Very short sentences are merged with a neighbour (a lone "First." sounds clipped) and a sentence longer than
    `max_chars` is cut at clause ends.
    """
    out: list[tuple[str, int]] = []
    paragraphs = [p for p in (" ".join(block.split()) for block in _PARAGRAPH.split(text)) if p]
    for para in paragraphs:
        merged: list[str] = []
        for sentence in (s for s in _SENTENCE_END.split(para) if s.strip()):
            if merged and len(merged[-1]) < MIN_SENTENCE_CHARS and len(merged[-1]) + 1 + len(sentence) <= max_chars:
                merged[-1] = f"{merged[-1]} {sentence}"
            else:
                merged.append(sentence)
        if (len(merged) > 1 and len(merged[-1]) < MIN_SENTENCE_CHARS
                and len(merged[-2]) + 1 + len(merged[-1]) <= max_chars):
            merged[-2:] = [f"{merged[-2]} {merged[-1]}"]
        for si, sentence in enumerate(merged):
            pieces = split_long_text(sentence, max_chars)
            for ki, piece in enumerate(pieces):
                pause = (CLAUSE_PAUSE_MS if ki < len(pieces) - 1
                         else SENTENCE_PAUSE_MS if si < len(merged) - 1 else PARAGRAPH_PAUSE_MS)
                out.append((piece, pause))
    if out:
        out[-1] = (out[-1][0], 0)
    return out


def split_long_text(text: str, max_chars: int) -> list[str]:
    """Split at sentence ends (then clause ends, then spaces) so each chunk fits `max_chars`."""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    for pattern in (_SENTENCE_END, _CLAUSE_END, re.compile(r"\s+")):
        pieces = pattern.split(text)
        if len(pieces) > 1:
            break
    else:
        return [text]  # a single unbreakable word
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    out: list[str] = []
    for chunk in chunks:  # a sentence may still be too long: split it by clauses / words
        out += [chunk] if len(chunk) <= max_chars else split_long_text(chunk, max_chars)
    return out


MAX_SEGMENTS = 60
MAX_SENTENCE_SEGMENTS = 250  # sentence-by-sentence engines: about the same text length as 60 chunks of 300
NEUTRAL_INTENSITY = 50

_EMOTION_EN = {
    "neutral": "neutral", "happy": "happy", "sad": "sad", "angry": "angry", "excited": "excited", "calm": "calm",
    "serious": "serious", "mysterious": "mysterious", "surprised": "surprised", "friendly": "warm and friendly",
    "professional": "professional",
}


@dataclass
class PlannedSegment:
    index: int
    text: str
    emotion: str | None
    intensity: int
    emphasis: bool = False
    whisper: bool = False
    pitch: str | None = None
    pause_before_ms: int = 0
    pause_after_ms: int = 0
    instruction: str | None = None  # style instruction for instruction-capable engines
    emotion_via: str | None = None  # "instruction" | "reference" | None


@dataclass
class GenerationPlan:
    segments: list[PlannedSegment]
    warnings: list[str] = field(default_factory=list)
    has_markup: bool = False

    @property
    def is_simple(self) -> bool:
        """A single segment without pauses or style: can use the plain generation path."""
        if len(self.segments) != 1:
            return False
        s = self.segments[0]
        return not (s.pause_before_ms or s.pause_after_ms or s.instruction or s.emotion_via)


class PlanError(ValueError):
    pass


def intensity_word(intensity: int) -> str:
    return "slightly " if intensity < 34 else "very " if intensity > 66 else ""


def build_instruction(seg: PlannedSegment) -> str | None:
    parts: list[str] = []
    if seg.emotion and seg.emotion != "neutral":
        parts.append(f"Speak in a {intensity_word(seg.intensity)}{_EMOTION_EN[seg.emotion]} tone.")
    elif seg.emotion == "neutral":
        parts.append("Speak in a neutral, even tone.")
    if seg.whisper:
        parts.append("Whisper softly.")
    if seg.emphasis:
        parts.append("Stress these words with clear emphasis.")
    if seg.pitch == "low":
        parts.append("Use a lower, deeper pitch.")
    elif seg.pitch == "high":
        parts.append("Use a higher pitch.")
    return " ".join(parts) or None


class _Warnings(list):
    def once(self, message: str) -> None:
        if message not in self:
            self.append(message)


def plan_generation(parsed: ParsedMarkup, caps: EngineCapabilities, engine_name: str,
                    default_emotion: str | None, intensity: int,
                    reference_emotions: set[str] | None = None) -> GenerationPlan:
    warnings = _Warnings()
    for warning in parsed.warnings:  # ElevenLabs tags with no equivalent, reported by the parser
        warnings.once(warning)
    controls = caps.controls
    emotion_src = controls[GenericControl.EMOTION].source
    pitch_src = controls[GenericControl.PITCH].source
    instruction_ok = controls[GenericControl.INSTRUCTION].source == ControlSource.NATIVE
    reference_emotions = reference_emotions or set()

    def effective(style: Style) -> tuple[str | None, bool, bool, str | None, str | None]:
        emotion = style.emotion or default_emotion
        via = None
        if emotion:
            label = EMOTIONS[emotion].lower()
            if emotion_src == ControlSource.INSTRUCTION and instruction_ok:
                via = "instruction"
            elif emotion_src == ControlSource.SEGMENTATION:
                if emotion in reference_emotions:
                    via = "reference"
                    if intensity != NEUTRAL_INTENSITY:
                        warnings.once(f"{engine_name} aplica la emoción cambiando de referencia: la intensidad no se "
                                      "puede ajustar.")
                elif emotion != "neutral":
                    warnings.once(f"No hay una referencia etiquetada como «{label}» en el perfil de voz: ese fragmento "
                                  "usará la referencia por defecto.")
            elif emotion != "neutral":
                warnings.once(f"{engine_name} no puede aplicar emociones: se ignora «{label}».")
            if via is None:
                emotion = None
        whisper = style.whisper and instruction_ok
        if style.whisper and not instruction_ok:
            warnings.once(f"{engine_name} no admite susurro: se ignora [susurro].")
        emphasis = style.emphasis and instruction_ok
        if style.emphasis and not instruction_ok:
            warnings.once(f"{engine_name} no admite énfasis: se ignora [énfasis].")
        pitch = style.pitch if style.pitch in ("low", "high") else None
        if pitch and not (pitch_src == ControlSource.INSTRUCTION and instruction_ok):
            warnings.once("Este motor no cambia el tono con marcas: se ignora [tono]. Puedes ajustarlo en "
                          "«Posprocesado».")
            pitch = None
        return emotion, whisper, emphasis, pitch, via

    segments: list[PlannedSegment] = []
    pending_pause = 0
    for event in parsed.events:
        if isinstance(event, Pause):
            pending_pause += event.ms
        elif isinstance(event, Sound):
            if event.kind == "breath":
                warnings.once("Ningún motor actual genera respiraciones: [respira] se sustituye por una pausa breve.")
                pending_pause += BREATH_PAUSE_MS
            else:
                name = "risa" if event.kind == "laugh" else "suspiro"
                warnings.once(f"Ningún motor actual genera sonidos no verbales: se ignora [{name}].")
        elif isinstance(event, TextRun):
            emotion, whisper, emphasis, pitch, via = effective(event.style)
            text = event.text
            last = segments[-1] if segments else None
            same_style = last is not None and (last.emotion, last.whisper, last.emphasis, last.pitch) == (
                emotion, whisper, emphasis, pitch)
            if last is not None and same_style and pending_pause == 0:
                last.text += text
                continue
            if not text.strip():
                continue
            if last is None:
                seg = PlannedSegment(0, text, emotion, intensity, emphasis, whisper, pitch,
                                     pause_before_ms=pending_pause, emotion_via=via)
            else:
                last.pause_after_ms += pending_pause
                seg = PlannedSegment(len(segments), text, emotion, intensity, emphasis, whisper, pitch, emotion_via=via)
            pending_pause = 0
            segments.append(seg)
    if segments:
        segments[-1].pause_after_ms += pending_pause

    if caps.sentence_chunks:
        split: list[PlannedSegment] = []
        for i, seg in enumerate(segments):
            chunks = sentence_chunks(seg.text, caps.max_chars_per_call or 10_000)
            for k, (part, pause) in enumerate(chunks):
                last = k == len(chunks) - 1
                split.append(PlannedSegment(
                    0, part, seg.emotion, seg.intensity, seg.emphasis, seg.whisper, seg.pitch,
                    pause_before_ms=seg.pause_before_ms if k == 0 else 0,
                    pause_after_ms=seg.pause_after_ms if last else pause, emotion_via=seg.emotion_via))
            if chunks and i < len(segments) - 1 and seg.pause_after_ms == 0:
                # a style change at a sentence or paragraph end still gets its natural pause
                if _ENDS_PARAGRAPH.search(seg.text) or _STARTS_PARAGRAPH.match(segments[i + 1].text):
                    split[-1].pause_after_ms = PARAGRAPH_PAUSE_MS
                elif seg.text.rstrip().endswith((".", "!", "?", "…")):
                    split[-1].pause_after_ms = SENTENCE_PAUSE_MS
        segments = split
    elif caps.max_chars_per_call:
        split = []
        for seg in segments:
            text = " ".join(seg.text.split())
            parts = split_long_text(text, caps.max_chars_per_call)
            if len(parts) > 1:
                warnings.once(f"El texto es largo para {engine_name}: se genera por frases (máximo "
                              f"{caps.max_chars_per_call} caracteres por fragmento) para avanzar más rápido y poder "
                              "cancelar entre fragmentos.")
            for k, part in enumerate(parts):
                split.append(PlannedSegment(
                    0, part, seg.emotion, seg.intensity, seg.emphasis, seg.whisper, seg.pitch,
                    pause_before_ms=seg.pause_before_ms if k == 0 else 0,
                    pause_after_ms=seg.pause_after_ms if k == len(parts) - 1 else 0, emotion_via=seg.emotion_via))
        segments = split

    cleaned: list[PlannedSegment] = []
    for seg in segments:
        seg.text = " ".join(seg.text.split())
        if not any(ch.isalnum() for ch in seg.text):
            if cleaned:  # punctuation-only leftovers: keep their pauses
                cleaned[-1].pause_after_ms += seg.pause_before_ms + seg.pause_after_ms
            continue
        seg.text, shouted = soften_shouting(seg.text)
        if shouted:
            warnings.once("El texto en MAYÚSCULAS se envía en minúsculas: los modelos no lo interpretan como énfasis y "
                          "pueden pronunciarlo mal. Para dar fuerza usa signos de exclamación o [emoción:…].")
        seg.index = len(cleaned)
        if seg.emotion_via == "instruction" or seg.whisper or seg.emphasis or seg.pitch:
            seg.instruction = build_instruction(seg)
        cleaned.append(seg)

    if not cleaned:
        raise PlanError("No hay texto que generar.")
    limit = MAX_SENTENCE_SEGMENTS if caps.sentence_chunks else MAX_SEGMENTS
    if len(cleaned) > limit:
        raise PlanError(f"El texto genera {len(cleaned)} segmentos; el máximo es {limit}.")
    return GenerationPlan(segments=cleaned, warnings=list(warnings), has_markup=parsed.has_markup)
