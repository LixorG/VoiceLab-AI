"""Prepare a script for speech without changing its words.

Rule (the same one ElevenLabs gives for its audio tags): the words of the script are never changed. Only
punctuation, sentence and paragraph breaks, formatting (markdown, emojis, list bullets, hashtag signs) and tags are
touched. ElevenLabs v3 audio tags and v2 ``<break>`` tags are translated to SpeechMarkup when VoiceLab has an
equivalent, and removed (with a note) when it does not, so a script written for ElevenLabs can be pasted as is.
Anything that would need a different word (a number read in another language, a list number that will be spoken,
an abbreviation) is reported as an alert and left to the user. Long sentences get a proposed cut that is only
applied if the user accepts it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from app.generation.markup import (
    EL_EMOTIONS,
    EL_EMPHASIS,
    EL_IGNORED,
    EL_NEXT_WORD,
    EL_PAUSES,
    EL_SOUNDS,
    EL_WHISPER,
)
from app.voice_profiles.emotions import EMOTIONS

_SPANISH_SOUND = {"breath": "respira", "laugh": "risa", "sigh": "suspiro"}

LONG_SENTENCE_WORDS = 20  # the ElevenLabs guidance aims at 8–14 words per sentence for a conversational delivery
SHORT_SCRIPT_WORDS = 40  # below this, results vary more between generations (ElevenLabs note)
LONG_SCRIPT_WORDS = 250  # ≈ 90 s: the format holds, the audience's attention may not
LONG_PARAGRAPH_WORDS = 120  # one block that long sounds like reading: no longer pause between ideas
CHARS_PER_SECOND = 14.0  # typical speech rate, for the estimate only
SENTENCE_PAUSE_S, PARAGRAPH_PAUSE_S = 0.35, 0.75

_SENTENCE_END = ".!?…"
_CLOSERS = "\"'»”’)]"


@dataclass
class Change:
    kind: str  # formato | etiqueta | puntuación | párrafo | emoji
    before: str
    after: str
    reason: str


@dataclass
class Alert:
    kind: str  # idioma | lista | mayúsculas | símbolo | enlace | paréntesis | números | longitud | etiqueta | pausa
    message: str
    excerpt: str | None = None


@dataclass
class Suggestion:
    before: str
    after: str
    reason: str


@dataclass
class Stats:
    words: int
    sentences: int
    paragraphs: int
    average_words: float
    longest_words: int
    seconds: float


@dataclass
class Prepared:
    text: str
    changes: list[Change] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    suggestions: list[Suggestion] = field(default_factory=list)
    stats: Stats | None = None


# ---------------------------------------------------------------------------------------------------- tags
_VOICELAB_TAG = re.compile(
    r"^/?\s*(pausa|pause|respira|respiraci[oó]n|breath|risa|laugh|suspiro|sigh|[eé]nfasis|emphasis|susurro|whisper|"
    r"tono|pitch|emoci[oó]n|emotion)\s*(:.*)?$", re.IGNORECASE)
_TAG = re.compile(r"(?<!\\)\[([^\[\]\n]{1,40})\]|<break\s+time\s*=\s*[\"']?([\d.,]+)\s*(ms|s)?[\"']?\s*/?>",
                  re.IGNORECASE)


def _is_voicelab(key: str, text: str) -> bool:
    if not _VOICELAB_TAG.match(key) or key == "pause":  # [pause] without a duration is the ElevenLabs tag
        return False
    if key in ("whisper", "emphasis"):  # ElevenLabs style has no closing tag; VoiceLab spans do
        return f"[/{key}]" in text.lower()
    return True


def _sentence_end(text: str, start: int) -> int:
    """Index where the sentence that starts at `start` ends (after its final punctuation), or the line end."""
    match = re.compile(rf"[{re.escape(_SENTENCE_END)}]+[{re.escape(_CLOSERS)}]*|\n").search(text, start)
    if match is None:
        return len(text)
    return match.start() if match.group(0) == "\n" else match.end()


def convert_tags(text: str, changes: list[Change], alerts: list[Alert]) -> str:
    out: list[str] = []
    cursor = 0
    dropped: list[str] = []
    pending_close: list[tuple[int, str]] = []  # (position in the source text, closing tag)

    matches = list(_TAG.finditer(text))
    for i, match in enumerate(matches):
        # close spans that end before this tag
        while pending_close and pending_close[0][0] <= match.start():
            pos, close = pending_close.pop(0)
            out.append(text[cursor:pos] + close)
            cursor = pos
        out.append(text[cursor:match.start()])
        cursor = match.end()
        if match.group(2) is not None:  # <break time="1.5s" />
            number = float(match.group(2).replace(",", "."))
            ms = int(round(number if (match.group(3) or "s").lower() == "ms" else number * 1000))
            ms = max(1, min(ms, 10_000))
            new = f"[pausa:{ms}ms]"
            out.append(new)
            changes.append(Change("etiqueta", match.group(0), new, "Las pausas de ElevenLabs v2 se escriben así en "
                                                                     "VoiceLab."))
            continue
        raw = match.group(1)
        key = re.sub(r"\s+", " ", raw.strip().lower())
        if _is_voicelab(key, text):
            out.append(match.group(0))  # already SpeechMarkup
            continue
        next_tag = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        end = min(_sentence_end(text, cursor), next_tag)
        if key in EL_PAUSES:
            new = f"[pausa:{EL_PAUSES[key]}ms]"
            reason = "Pausa de ElevenLabs traducida a una pausa de VoiceLab."
        elif key in EL_SOUNDS:
            new = f"[{_SPANISH_SOUND[EL_SOUNDS[key]]}]"
            reason = ("Sonido no verbal traducido; los motores actuales no lo generan (la respiración se convierte "
                      "en una pausa breve y el resto se ignora al generar).")
        elif key in EL_WHISPER or key == "whisper":
            new, reason = "[susurro]", "Susurro hasta el final de la frase."
            pending_close.append((end, "[/susurro]"))
        elif key in EL_EMPHASIS or key == "emphasis":
            new, reason = "[énfasis]", "Énfasis hasta el final de la frase."
            pending_close.append((end, "[/énfasis]"))
        elif key in EL_NEXT_WORD:
            word = re.compile(r"\s*\S+").match(text, cursor)
            if word is None:
                dropped.append(raw)
                continue
            new, reason = "[énfasis]", "Énfasis en la palabra siguiente."
            pending_close.append((word.end(), "[/énfasis]"))
        elif key in EL_EMOTIONS:
            label = EMOTIONS[EL_EMOTIONS[key]].lower()
            new, reason = f"[emoción:{label}]", "Emoción hasta el final de la frase (se aplica si el motor puede)."
            pending_close.append((end, "[/emoción]"))
        else:  # documented but with no equivalent, or something the parser would reject
            dropped.append(raw)
            if key not in EL_IGNORED:
                alerts.append(Alert("etiqueta", "Etiqueta desconocida: se quita para que el texto se pueda generar.",
                                    f"[{raw.strip()}]"))
            continue
        pending_close.sort()
        out.append(new)
        changes.append(Change("etiqueta", match.group(0), new, reason))
    while pending_close:
        pos, close = pending_close.pop(0)
        out.append(text[cursor:pos] + close)
        cursor = pos
    out.append(text[cursor:])
    if dropped:
        names = ", ".join(f"[{d}]" for d in dict.fromkeys(dropped))
        changes.append(Change("etiqueta", names, "", "Sin equivalente en VoiceLab: se quitan (no son palabras)."))
        alerts.append(Alert("etiqueta", "Estas etiquetas no tienen equivalente en los motores de VoiceLab y se "
                                        "quitaron; el ritmo tendrá que salir de la puntuación.", names))
    return "".join(out)


# ---------------------------------------------------------------------------------------------------- formatting
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF⬀-⯿️‍]+")
_ITEM = "\u2063"  # invisible marker for a list item while formatting (removed right after)
_BULLET = re.compile(r"^(\s*)[-*•·▪►✓✔]\s+(?=\S)", re.MULTILINE)
_NUMBERED = re.compile(r"^\s*(\d{1,2})[.)]\s+(?=\S)", re.MULTILINE)
_HEADING = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
_BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_ITALIC = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])")
_CODE = re.compile(r"`([^`\n]+)`")
_HASHTAG = re.compile(r"(?<![\w&])#([^\W\d_][\w]*)")


def _record(pattern: re.Pattern, repl, text: str, changes: list[Change], kind: str, reason: str) -> str:
    def sub(m: re.Match) -> str:
        new = repl(m) if callable(repl) else m.expand(repl)
        if new != m.group(0):
            changes.append(Change(kind, m.group(0), new, reason))
        return new

    return pattern.sub(sub, text)


def clean_format(text: str, changes: list[Change]) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = _record(_HEADING, lambda m: _ITEM, text, changes, "formato",
                   "Marca de título (markdown): no se lee; el título cierra su propia frase.")
    text = _record(_BULLET, lambda m: m.group(1) + _ITEM, text, changes, "formato",
                   "Viñeta de lista: no se lee; cada punto va en su propia frase.")
    lines = text.split("\n")
    for k, line in enumerate(lines):
        if line.startswith(_ITEM):
            item = line[len(_ITEM):]
            if not _ends_sentence(item) and any(ch.isalnum() for ch in item):
                changes.append(Change("puntuación", item, _add_period(item), "Cada punto de la lista cierra su frase."))
                item = _add_period(item)
            lines[k] = item
    text = "\n".join(lines)
    for change in changes:
        change.after = change.after.replace(_ITEM, "")
    text = _record(_BOLD, r"\2", text, changes, "formato", "Negrita (markdown): no cambia la voz.")
    text = _record(_ITALIC, r"\1", text, changes, "formato", "Cursiva (markdown): no cambia la voz.")
    text = _record(_CODE, r"\1", text, changes, "formato", "Comillas de código: no se leen.")
    text = _record(_HASHTAG, r"\1", text, changes, "formato", "Almohadilla de hashtag: se leería como símbolo.")
    emojis = _EMOJI.findall(text)
    if emojis:
        text = _EMOJI.sub("", text)
        changes.append(Change("emoji", " ".join(dict.fromkeys(emojis)), "", "Los emojis no se leen (o se leen mal)."))
    return text


_SPACES_BEFORE = re.compile(r"[ \xa0]+([,.;:!?…])")  # «50 %» (Spanish style) is left alone
_MISSING_SPACE = re.compile(r"([^\W\d_]{2}[.!?…]|[,;:])([^\W\d_¿¡])")
_LINK = re.compile(r"https?://\S+|www\.\S+|[\w.+-]+@[\w-]+\.[\w.]+|\b[\w-]+\.(?:com|org|net|es|co|io|app|dev)\b\S*")


_PROTECTED = re.compile(rf"{_LINK.pattern}|(?<!\\)\[[^\[\]\n]{{1,40}}\]")


def clean_punctuation(text: str, changes: list[Change]) -> str:
    """Punctuation fixes outside links (the period in «ejemplo.com» does not end a sentence)."""
    parts, cursor = [], 0
    for link in _PROTECTED.finditer(text):
        parts.append(_clean_punctuation(text[cursor:link.start()], changes))
        parts.append(link.group(0))
        cursor = link.end()
    parts.append(_clean_punctuation(text[cursor:], changes))
    return "\n".join(line.strip() for line in "".join(parts).split("\n"))


def _clean_punctuation(text: str, changes: list[Change]) -> str:
    text = _record(re.compile(r"(?<!\.)\.\s?\.(?:\s?\.)*|…{2,}"), "…", text, changes, "puntuación",
                   "Puntos suspensivos como un solo signo: pausa con peso, siempre igual.")
    text = _record(re.compile(r"!{2,}"), "!", text, changes, "puntuación",
                   "Un signo basta; repetirlo no da más énfasis.")
    text = _record(re.compile(r"\?{2,}"), "?", text, changes, "puntuación", "Un signo basta.")
    text = _record(_SPACES_BEFORE, r"\1", text, changes, "puntuación", "Sin espacio antes del signo.")
    text = _record(_MISSING_SPACE, r"\1 \2", text, changes, "puntuación", "Falta el espacio después del signo.")
    return re.sub(r"[ \xa0]{2,}", " ", text)


def _ends_sentence(line: str) -> bool:
    bare = re.sub(r"\[[^\[\]]*\]\s*$", "", line).rstrip()
    while bare.endswith(("[/emoción]", "[/énfasis]", "[/susurro]", "[/tono]")):
        bare = bare[:bare.rindex("[")].rstrip()
    return bare.rstrip(_CLOSERS).endswith(tuple(_SENTENCE_END + ":;,"))


def fix_paragraphs(text: str, changes: list[Change]) -> str:
    """Join hard-wrapped lines, keep deliberate line breaks, close lines that have no final punctuation."""
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    out_blocks = []
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        merged: list[str] = []
        for line in lines:
            if merged and not _ends_sentence(merged[-1]) and line[:1].islower():
                changes.append(Change("párrafo", f"{merged[-1][-30:]}⏎{line[:30]}", f"{merged[-1][-30:]} {line[:30]}",
                                      "Línea cortada a mitad de frase: se une."))
                merged[-1] = f"{merged[-1]} {line}"
            else:
                merged.append(line)
        for k, line in enumerate(merged):
            if not _ends_sentence(line) and any(ch.isalnum() for ch in line):
                changes.append(Change("puntuación", line[-40:], f"{line[-40:]}.",
                                      "Sin punto final la entonación queda en el aire y la frase se funde con la "
                                      "siguiente."))
                merged[k] = _add_period(line)
        out_blocks.append("\n".join(merged))
    return "\n\n".join(out_blocks)


def _add_period(line: str) -> str:
    """Put the period before trailing closing tags ([/emoción]) so the tag keeps wrapping the whole sentence."""
    match = re.search(r"((?:\[/[^\[\]]+\]|\[[^\[\]]*\])\s*)+$", line)
    if match:
        return line[:match.start()].rstrip() + "." + line[match.start():]
    return line + "."


# ---------------------------------------------------------------------------------------------------- analysis
_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_STRIP_TAGS = re.compile(r"(?<!\\)\[[^\[\]\n]{1,40}\]")
_CONNECTORS = {
    "en": ("and", "but", "so", "because", "then", "which", "while", "although"),
    "es": ("y", "pero", "así que", "porque", "entonces", "aunque", "mientras"),
}
_NUMBER_WORDS = {
    "es": {"uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "veinte", "cien", "mil"},
    "en": {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "twenty", "hundred",
           "thousand"},
}


def sentences(text: str) -> list[str]:
    plain = _STRIP_TAGS.sub("", text)
    parts = re.split(rf"(?<=[{re.escape(_SENTENCE_END)}])[{re.escape(_CLOSERS)}]*\s+|\n+", plain)
    return [p.strip() for p in parts if _WORD.search(p)]


def split_suggestions(text: str, language: str | None) -> list[Suggestion]:
    """For sentences over LONG_SENTENCE_WORDS words, cut at the connector (or semicolon) closest to the middle."""
    connectors = _CONNECTORS.get(language or "", ()) or _CONNECTORS["en"] + _CONNECTORS["es"]
    pattern = re.compile(r"(,|;)\s+(" + "|".join(re.escape(c) for c in connectors) + r")\b\s*|;\s+", re.IGNORECASE)
    out = []
    for sentence in sentences(text):
        words = _WORD.findall(sentence)
        if len(words) <= LONG_SENTENCE_WORDS or sentence not in text:
            continue
        cuts = [m for m in pattern.finditer(sentence)
                if 4 <= len(_WORD.findall(sentence[:m.start()])) and 4 <= len(_WORD.findall(sentence[m.end():]))]
        if not cuts:
            continue
        middle = len(sentence) / 2
        cut = min(cuts, key=lambda m: abs(m.start() - middle))
        connector = cut.group(2)
        rest = sentence[cut.end():]
        if connector:
            second = connector[0].upper() + connector[1:] + " " + rest
        else:
            second = rest[:1].upper() + rest[1:]
        after = sentence[:cut.start()] + ". " + second
        out.append(Suggestion(sentence, after, f"{len(words)} palabras en una sola frase: suena a lectura. Partida "
                                               "en dos, cada idea tiene su entonación."))
    return out


def analyse(text: str, language: str | None, alerts: list[Alert]) -> Stats:
    plain = _STRIP_TAGS.sub("", text)
    sents = sentences(text)
    counts = [len(_WORD.findall(s)) for s in sents]
    words = sum(counts)
    paragraphs = len([b for b in re.split(r"\n\s*\n", text) if b.strip()])
    seconds = len(plain.replace("\n", " ").strip()) / CHARS_PER_SECOND + max(0, len(sents) - paragraphs) * \
        SENTENCE_PAUSE_S + max(0, paragraphs - 1) * PARAGRAPH_PAUSE_S
    if 0 < words < SHORT_SCRIPT_WORDS:
        alerts.append(Alert("longitud", f"Guion corto ({words} palabras): con menos de {SHORT_SCRIPT_WORDS} los "
                                        "resultados varían más entre generaciones. Genera varias variaciones."))
    elif words > LONG_SCRIPT_WORDS:
        alerts.append(Alert("longitud", f"{words} palabras (≈ {seconds / 60:.1f} min): el formato aguanta, pero para "
                                        "vídeos cortos la atención suele caer pasado el minuto y medio."))

    other = {"en": "es", "es": "en"}.get(language or "")
    tokens = _WORD.findall(plain)
    if language == "en":
        foreign = [t for t in tokens if re.search(r"[áéíóúñÁÉÍÓÚÑ]", t) or t.lower() in _NUMBER_WORDS["es"]]
    elif other:
        foreign = [t for t in tokens if t.lower() in _NUMBER_WORDS[other]]
    else:
        foreign = []
    if foreign:
        names = ", ".join(dict.fromkeys(foreign))
        alerts.append(Alert("idioma", f"Posibles palabras en otro idioma dentro de un texto en "
                                      f"{'inglés' if language == 'en' else 'español'}: se leerán con la pronunciación "
                                      "del texto. Revísalas (no se cambian solas).", names))
    caps = [t for t in tokens if len(t) >= 3 and t.isupper()]
    if caps:
        alerts.append(Alert("mayúsculas", "Las MAYÚSCULAS no dan énfasis en estos motores (y a veces se leen letra a "
                                          "letra). Para remarcar, usa [énfasis]…[/énfasis], un signo de exclamación o "
                                          "una pausa antes.", ", ".join(dict.fromkeys(caps))))
    numbered = _NUMBERED.findall(text)
    if numbered:
        alerts.append(Alert("lista", "Los números de lista se leerán en voz alta («1.» → «one»/«uno»). Quítalos si "
                                     "no quieres que se digan; el orden ya se entiende por las frases.",
                            ", ".join(f"{n}." for n in numbered)))
    links = _LINK.findall(plain)
    if links:
        alerts.append(Alert("enlace", "Los enlaces y correos se leen carácter a carácter. Escríbelos como quieres "
                                      "que suenen («ejemplo punto com»).", ", ".join(links)))
    parens = re.findall(r"\([^()\n]{1,80}\)", plain)
    if parens:
        alerts.append(Alert("paréntesis", "Los paréntesis se leen en el mismo tono que el resto: si es un inciso, "
                                          "usa comas o una raya (—).", " ".join(parens)))
    symbols = sorted(set(re.findall(r"[&/@=+*<>|~^_\\]", _LINK.sub("", plain))))
    if symbols:
        alerts.append(Alert("símbolo", "Símbolos que cada motor lee a su manera (o se salta). Escríbelos con "
                                       "palabras si deben sonar.", " ".join(symbols)))
    if re.search(r"\n\s*\n\s*\[pausa:", text):
        alerts.append(Alert("pausa", "Una pausa al principio de un párrafo se suma a la pausa entre párrafos: dos "
                                     "silencios seguidos. Mejor dentro de la frase."))
    long_blocks = [b for b in re.split(r"\n\s*\n", text)
                   if len(_WORD.findall(_STRIP_TAGS.sub("", b))) > LONG_PARAGRAPH_WORDS]
    if long_blocks:
        alerts.append(Alert("párrafo", f"{len(long_blocks)} párrafo(s) de más de {LONG_PARAGRAPH_WORDS} palabras. "
                                       "Deja una línea en blanco donde cambia la idea: ahí la voz hace una pausa más "
                                       "larga y el oyente respira.", long_blocks[0][:60] + "…"))
    return Stats(words=words, sentences=len(sents), paragraphs=paragraphs,
                 average_words=round(words / len(sents), 1) if sents else 0.0,
                 longest_words=max(counts, default=0), seconds=round(seconds, 1))


def prepare(text: str, language: str | None) -> Prepared:
    changes: list[Change] = []
    alerts: list[Alert] = []
    out = clean_format(text, changes)
    out = convert_tags(out, changes, alerts)
    out = clean_punctuation(out, changes)
    out = fix_paragraphs(out, changes)
    out = unicodedata.normalize("NFC", out).strip()
    stats = analyse(out, language, alerts)
    return Prepared(text=out, changes=changes, alerts=alerts, suggestions=split_suggestions(out, language),
                    stats=stats)


def same_words(a: str, b: str) -> bool:
    """True when two texts have the same words in the same order (tags and punctuation aside)."""
    def words(text: str) -> list[str]:
        text = _TAG.sub(" ", _EMOJI.sub(" ", text))
        return [w.lower() for w in _WORD.findall(text)]

    return words(a) == words(b)
