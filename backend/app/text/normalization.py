"""Text normalisation for TTS: write numbers, symbols and abbreviations the way they are said.

TTS models read digits and abbreviations inconsistently ("100%", "Sr.", "10:30" can come out spelled, skipped or
in another language). Everything here is deterministic and reported as a list of changes, so the UI can show the
exact text sent to the engine. Only Spanish and English are normalised; other languages are left untouched.

Order: user dictionary (it wins, e.g. «IA» → «i a») → abbreviations → numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SUPPORTED_LANGUAGES = ("es", "en")


@dataclass(frozen=True)
class DictionaryEntry:
    term: str
    replacement: str
    case_sensitive: bool = False


@dataclass(frozen=True)
class TextChange:
    original: str
    replacement: str
    kind: str  # "diccionario" | "abreviatura" | "número"


# ---------------------------------------------------------------- abbreviations
_ABBREVIATIONS = {
    "es": {
        "Sr.": "señor", "Sra.": "señora", "Srta.": "señorita", "Sres.": "señores", "Dr.": "doctor", "Dra.": "doctora",
        "Lic.": "licenciado", "Ing.": "ingeniero", "Prof.": "profesor", "Profa.": "profesora", "Ud.": "usted",
        "Uds.": "ustedes", "etc.": "etcétera", "p. ej.": "por ejemplo", "EE. UU.": "Estados Unidos",
        "EE.UU.": "Estados Unidos", "aprox.": "aproximadamente", "pág.": "página", "págs.": "páginas",
        "núm.": "número", "tel.": "teléfono", "vs.": "contra",
    },
    "en": {
        "Mr.": "mister", "Mrs.": "missus", "Ms.": "miz", "Dr.": "doctor", "Prof.": "professor", "etc.": "et cetera",
        "e.g.": "for example", "i.e.": "that is", "vs.": "versus", "approx.": "approximately",
    },
}
# Titles go before a name ("Sr. Pérez"), so their dot never ends a sentence.
_TITLES = {"señor", "señora", "señorita", "señores", "doctor", "doctora", "licenciado", "ingeniero", "profesor",
           "profesora", "mister", "missus", "miz", "professor"}
# Units are only expanded right after a number ("5 km"), where they cannot be ordinary words.
_UNITS = {
    "es": {"km": "kilómetros", "m": "metros", "cm": "centímetros", "mm": "milímetros", "kg": "kilos", "g": "gramos",
           "l": "litros", "ml": "mililitros", "h": "horas", "min": "minutos", "s": "segundos", "GB": "gigas",
           "MB": "megas", "°C": "grados", "km/h": "kilómetros por hora"},
    "en": {"km": "kilometers", "m": "meters", "cm": "centimeters", "kg": "kilograms", "g": "grams", "h": "hours",
           "min": "minutes", "s": "seconds", "GB": "gigabytes", "MB": "megabytes", "°C": "degrees",
           "km/h": "kilometers per hour"},
}
_CURRENCY = {
    # "$" means pesos in most Spanish-speaking countries; write "US$" or "USD" for dollars.
    "es": {"US$": "dólares", "USD": "dólares", "$": "pesos", "€": "euros", "EUR": "euros", "£": "libras",
           "COP": "pesos", "MXN": "pesos"},
    "en": {"US$": "dollars", "USD": "dollars", "$": "dollars", "€": "euros", "EUR": "euros", "£": "pounds"},
}
_WORDS = {
    "es": {"percent": "por ciento", "decimal": "coma", "and": "y", "o_clock": "en punto", "minus": "menos",
           "with": "con", "cents": "centavos"},
    "en": {"percent": "percent", "decimal": "point", "and": "", "o_clock": "o'clock", "minus": "minus",
           "with": "and", "cents": "cents"},
}
_NEG = r"(?P<neg>(?<![\w-])-)?"


def _boundary(term: str) -> str:
    """Regex for a term that must not be glued to other letters/digits (terms may contain spaces or dots)."""
    return rf"(?<![\w]){re.escape(term)}(?![\w])"


# ---------------------------------------------------------------- numbers
def _int_words(value: int, language: str, ordinal: bool = False, feminine: bool = False) -> str:
    from num2words import num2words

    words = num2words(value, lang=language, to="ordinal" if ordinal else "cardinal").replace(",", "")
    if language == "es" and ordinal and feminine and words.endswith("o"):
        words = words[:-1] + "a"
    return words


def _apocope(words: str) -> str:
    """Spanish: «veintiuno años» → «veintiún años», «uno días» → «un días» (masculine noun follows)."""
    if words.endswith("veintiuno"):
        return words[:-len("veintiuno")] + "veintiún"
    if words.endswith("uno"):
        return words[:-3] + "un"
    return words


def _parse_number(raw: str, language: str) -> tuple[int, str | None] | None:
    """Returns (integer part, decimal digits) honouring each language's separators; None if ambiguous/invalid."""
    thousands, decimal = (".", ",") if language == "es" else (",", ".")
    if re.fullmatch(rf"\d{{1,3}}(?:{re.escape(thousands)}\d{{3}})+(?:{re.escape(decimal)}\d+)?", raw) or \
            re.fullmatch(rf"\d+(?:{re.escape(decimal)}\d+)?", raw):
        integer, _, frac = raw.replace(thousands, "").partition(decimal)
        return int(integer), (frac or None)
    return None


def _number_words(raw: str, language: str) -> str | None:
    parsed = _parse_number(raw, language)
    if parsed is None:
        return None
    integer, frac = parsed
    words = _int_words(integer, language)
    if frac is not None:
        # "3,05" → "tres coma cero cinco": leading zeros are said digit by digit, the rest as a number.
        zeros = len(frac) - len(frac.lstrip("0"))
        tail = frac.lstrip("0")
        parts = [_int_words(0, language)] * zeros + ([_int_words(int(tail), language)] if tail else [])
        words = f"{words} {_WORDS[language]['decimal']} {' '.join(parts)}"
    return words


_NUM = r"\d[\d.,]*\d|\d"


def _normalize_numbers(text: str, language: str, changes: list[TextChange]) -> str:
    w = _WORDS[language]

    def record(original: str, replacement: str) -> str:
        changes.append(TextChange(original.strip(), replacement, "número"))
        return replacement

    # currency before or after the amount: "$20.000", "20 €", "USD 15"
    symbols = sorted(_CURRENCY[language], key=len, reverse=True)
    sym = "|".join(re.escape(s) for s in symbols)

    def currency(m: re.Match[str]) -> str:
        amount, symbol = (m.group("a1"), m.group("s1")) if m.group("a1") else (m.group("a2"), m.group("s2"))
        parsed = _parse_number(amount, language)
        if parsed is None:
            return m.group(0)
        integer, frac = parsed
        unit = _CURRENCY[language][symbol]
        if frac is not None and len(frac) == 2:  # money: "12,50 €" → "doce euros con cincuenta centavos"
            main = _int_words(integer, language)
            main = _apocope(main) if language == "es" else main
            said = f"{main} {unit} {w['with']} {_int_words(int(frac), language)} {w['cents']}"
        else:
            words = _number_words(amount, language) or ""
            said = f"{_apocope(words) if language == 'es' else words} {unit}"
        return record(m.group(0), said)

    text = re.sub(rf"(?P<s1>{sym})\s?(?P<a1>{_NUM})|(?P<a2>{_NUM})\s?(?P<s2>{sym})(?![\w])", currency, text)

    def percent(m: re.Match[str]) -> str:
        words = _number_words(m.group("n"), language)
        sign = f"{w['minus']} " if m.group("neg") else ""
        return m.group(0) if words is None else record(m.group(0), f"{sign}{words} {w['percent']}")

    text = re.sub(rf"{_NEG}(?P<n>{_NUM})\s?%", percent, text)

    def clock(m: re.Match[str]) -> str:
        hours, minutes = int(m.group(1)), int(m.group(2))
        h = _int_words(hours, language)
        if minutes == 0:
            said = f"{h} {w['o_clock']}"
        else:
            said = f"{h} {w['and']} {_int_words(minutes, language)}".replace("  ", " ")
        return record(m.group(0), said)

    text = re.sub(r"(?<![\d:])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])", clock, text)

    if language == "es":
        def ordinal(m: re.Match[str]) -> str:
            return record(m.group(0), _int_words(int(m.group(1)), language, ordinal=True, feminine=m.group(2) == "ª"))

        text = re.sub(r"(?<!\d)(\d{1,3})\.?([ºª°])(?!\w)", ordinal, text)

    units = sorted(_UNITS[language], key=len, reverse=True)
    unit_re = "|".join(re.escape(u) for u in units)

    def with_unit(m: re.Match[str]) -> str:
        words = _number_words(m.group("n"), language)
        if words is None:
            return m.group(0)
        if language == "es":
            words = _apocope(words)
        sign = f"{w['minus']} " if m.group("neg") else ""
        return record(m.group(0), f"{sign}{words} {_UNITS[language][m.group('u')]}")

    text = re.sub(rf"{_NEG}(?P<n>{_NUM})\s?(?P<u>{unit_re})(?![\w/])", with_unit, text)

    def plain(m: re.Match[str]) -> str:
        raw = m.group("n")
        words = _number_words(raw, language)
        if words is None:
            return m.group(0)
        following = text[m.end():m.end() + 2]
        if language == "es" and re.match(r"\s[^\W\d_]", following):  # a noun follows: «21 años» → «veintiún años»
            words = _apocope(words)
        sign = f"{w['minus']} " if m.group("neg") else ""
        return record(m.group(0), sign + words)

    return re.sub(rf"{_NEG}(?<![\w.,])(?P<n>{_NUM})(?![\w.,]*\d)(?![\w])", plain, text)


# ---------------------------------------------------------------- public API
def normalize_text(text: str, language: str | None, dictionary: list[DictionaryEntry] | tuple = (),
                   numbers: bool = True) -> tuple[str, list[TextChange]]:
    """Apply the dictionary (any language), then abbreviations and numbers (Spanish/English)."""
    changes: list[TextChange] = []

    for entry in sorted(dictionary, key=lambda e: len(e.term), reverse=True):
        flags = 0 if entry.case_sensitive else re.IGNORECASE
        pattern = re.compile(_boundary(entry.term), flags)

        def swap(m: re.Match[str], entry: DictionaryEntry = entry) -> str:
            changes.append(TextChange(m.group(0), entry.replacement, "diccionario"))
            return entry.replacement

        text = pattern.sub(swap, text)

    if not numbers or language not in SUPPORTED_LANGUAGES:
        return text, changes

    for abbreviation, full in sorted(_ABBREVIATIONS[language].items(), key=lambda kv: len(kv[0]), reverse=True):
        def expand(m: re.Match[str], full: str = full, abbreviation: str = abbreviation) -> str:
            changes.append(TextChange(m.group(0), full, "abreviatura"))
            # The abbreviation's dot may also end the sentence ("… etc. Luego"): keep a full stop then.
            rest = m.string[m.end():].lstrip()
            ends_sentence = full not in _TITLES and (not rest or rest[0].isupper())
            return full + ("." if ends_sentence else "")

        text = re.sub(rf"(?<![\w]){re.escape(abbreviation)}(?=\s|$|[,;:!?)])", expand, text)

    return _normalize_numbers(text, language, changes), changes


def unique_changes(changes: list[TextChange]) -> list[TextChange]:
    seen: set[tuple[str, str]] = set()
    out = []
    for change in changes:
        key = (change.original, change.replacement)
        if key not in seen:
            seen.add(key)
            out.append(change)
    return out
