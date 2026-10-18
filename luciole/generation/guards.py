"""Gardes déterministes sur une réponse du LLM, reprises de ``query2`` (V5).

Aucune ne demande l'avis du LLM : ce sont des règles de code, testées, dont
les seuils viennent du profil (``GuardsSpec``).

- **esquive** : la réponse affirme tôt que le dossier ne dit rien, ou bien
  l'affirme et reste courte ou sans citation ;
- **dégénérée** : le décodage boucle (peu de lignes distinctes, ou une ligne
  répétée) ;
- **pages hors lot** : la réponse cite une page qu'aucun passage fourni ne
  couvre ;
- **nombres hors extraits** : une valeur chiffrée de la réponse n'existe dans
  aucun passage fourni. On la signale, on ne la « corrige » jamais.

Les pages du lot comprennent les pages PDF et les numéros imprimés
(« 14/47 » → 14) : les études d'impact impriment deux pages par page PDF, et
la réponse cite les numéros que le lecteur voit.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from luciole.retrieval.lot import strip_accents

# Motifs d'absence, en ASCII : le texte comparé est désaccentué.
_EVASION = re.compile(
    r"n(?:e |')(?:contien|mentionn|fourni|permet|cite|precise|indique|"
    r"detaille|comporte|evoque|abord)\w* (?:pas|aucun)|"
    r"aucune? (?:information|mention|precision|element)|"
    r"pas d(?:e |')(?:information|mention|precision)|"
    r"reste(?:nt)? muet|(?:est|sont) absente?s? d",
    re.I,
)
# Une citation : un numéro de page ou une étiquette de source.
_CITATION = re.compile(r"\bp(?:age)?s?\.?\s*\d+|source\s*:|\.pdf", re.I)
# Page(s) citée(s) : « page 73 », « pages 104-106 », « p. 60 ».
_PAGE_CITED = re.compile(r"\b(?:pages?|p\.)\s*(\d{1,4})(?:\s*(?:-|–|à)\s*(\d{1,4}))?", re.I)
_FILE_NAME = re.compile(r"([\w\-()/]+(?:[ .][\w\-()/]+)*\.(?:pdf|docx?|md|txt))", re.I)
_TOME = re.compile(r"\btome\s*(\d+)", re.I)
_LIST_MARK = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")
_NUMBER = re.compile(r"(?<![\w/])(\d{1,3}(?:[   ]\d{3})+|\d+)(?:[.,](\d+))?(?![\w/])")
_CITATION_ZONE = re.compile(
    r"\[Source:[^\]]*\]|\((?:Source|Tome)[^)]*\)"
    r"|\b(?:pages?|p\.)\s*\d+(?:\s*(?:-|–|à)\s*\d+)?",
    re.I,
)
_LIST_NUMBER = re.compile(r"^\s*\d+[.)]\s", re.M)
_THOUSANDS = re.compile(r"[   ]")


def is_evasive(text: str, max_position: float = 0.15, min_length: int = 700) -> bool:
    """Vrai si la réponse esquive. Longueur et citation ne comptent que si un
    motif d'absence a été trouvé : une réponse sourcée qui concède un manque
    à la fin n'esquive pas."""
    if not text or not text.strip():
        return True
    flat = strip_accents(text)
    match = _EVASION.search(flat)
    if not match:
        return False
    return (
        match.start() / len(flat) < max_position
        or len(flat) < min_length
        or not _CITATION.search(flat)
    )


def is_degenerate(
    text: str, min_lines: int = 8, min_ratio: float = 0.6, max_repeats: int = 4
) -> bool:
    """Vrai si la réponse boucle : moins de ``min_ratio`` lignes distinctes,
    ou une même ligne ``max_repeats`` fois (numérotation ignorée)."""
    lines = []
    for raw in (text or "").splitlines():
        line = re.sub(r"\W+", " ", strip_accents(_LIST_MARK.sub("", raw))).strip().lower()
        if len(line) > 20:
            lines.append(line)
    if len(lines) < min_lines:
        return False
    counts = Counter(lines)
    return len(counts) / len(lines) < min_ratio or counts.most_common(1)[0][1] >= max_repeats


# --- pages ----------------------------------------------------------------
@dataclass(frozen=True)
class SourcePages:
    """Ce qu'un passage fourni au LLM couvre : fichier (chemin dans la racine,
    ou nom), pages PDF, numéros imprimés (« 14/47 »)."""

    file_name: str
    page_start: int | None = None
    page_end: int | None = None
    page_labels: Sequence[str] = ()


def lot_pages(sources: Iterable[SourcePages]) -> dict[str, set[int]]:
    """Pages couvertes par fichier (chemin en minuscules) : pages PDF et
    numérateurs des numéros imprimés."""
    pages: dict[str, set[int]] = {}
    for s in sources:
        name = s.file_name.strip().lower()
        if not name:
            continue
        covered = pages.setdefault(name, set())
        if s.page_start is not None:
            end = s.page_end if s.page_end is not None else s.page_start
            covered.update(range(min(s.page_start, end), max(s.page_start, end) + 1))
        for label in s.page_labels:
            head = re.match(r"\s*(\d{1,4})", str(label))
            if head:
                covered.add(int(head.group(1)))
    return {name: covered for name, covered in pages.items() if covered}


def _cited_document(before: str, pages: Mapping[str, set[int]]) -> str | None:
    """Document d'une citation : la dernière référence avant la page, chemin
    ou nom de fichier (complet ou abrégé) ou « Tome N » ; la plus proche
    l'emporte. Un nom que portent plusieurs fichiers du lot (deux dossiers)
    ne désigne aucun d'eux : la page est alors comparée à tout le lot."""
    name, position = None, -1
    for cited in _FILE_NAME.finditer(before):
        words = cited.group(1).strip().split(" ")
        candidate = words[-1]
        for i in range(len(words)):
            token = " ".join(words[i:])
            matches = [n for n in pages if n == token or n.endswith(token)]
            if token in matches:
                matches = [token]
            if len(matches) > 1 and len({m.rsplit("/", 1)[-1] for m in matches}) == 1:
                candidate = None  # même nom dans plusieurs dossiers
                break
            if matches:
                candidate = min(matches, key=len)
                break
        name, position = candidate, cited.end()
    for tome in _TOME.finditer(before):
        if tome.end() > position:
            n = tome.group(1)
            keys = [k for k in pages if re.search(rf"tome[ _-]?0?{n}(?!\d)", k)]
            if len(keys) == 1:
                name, position = keys[0], tome.end()
    return name


def cited_pages(text: str) -> set[int]:
    """Numéros de page cités dans un texte (plages développées, au plus 21)."""
    pages: set[int] = set()
    for m in _PAGE_CITED.finditer(text or ""):
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        if end < start or end - start > 20:
            end = start
        pages.update(range(start, end + 1))
    return pages


def pages_outside_lot(text: str, pages: Mapping[str, set[int]], window: int = 300) -> list[str]:
    """Pages citées qu'aucun passage ne couvre, en libellés « fichier p. N ».
    Une page sans document reconnu est comparée à l'union des pages du lot."""
    if not text or not pages:
        return []
    union = set().union(*pages.values())
    outside: list[str] = []
    for m in _PAGE_CITED.finditer(text):
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        if end < start or end - start > 20:
            end = start
        before = text[max(0, m.start() - window) : m.start()].lower()
        name = _cited_document(before, pages)
        reference = pages.get(name, union) if name else union
        if end != start and start in reference and end in reference:
            continue  # plage dont les deux bornes sont dans le lot
        for page in range(start, end + 1):
            if page not in reference:
                label = f"{name} p. {page}" if name else f"p. {page}"
                if label not in outside:
                    outside.append(label)
    return outside


# --- nombres --------------------------------------------------------------
def _key(integer: str, decimals: str | None) -> str:
    integer = _THOUSANDS.sub("", integer)
    decimals = (decimals or "").rstrip("0")
    return f"{integer}.{decimals}" if decimals else integer


def numbers_of(text: str) -> set[str]:
    """Nombres normalisés : espaces de milliers retirés, virgule décimale en
    point, zéros de fin retirés (« 2 830 » → 2830, « 0,80 » → 0.8)."""
    return {_key(m.group(1), m.group(2)) for m in _NUMBER.finditer(text or "")}


def numbers_outside_extracts(answer: str, extracts: str, min_digits: int = 2) -> list[str]:
    """Nombres de la réponse absents des extraits, tels qu'écrits. Ignorés :
    citations de pages, numérotation de liste, entiers de moins de
    ``min_digits`` chiffres (trop ambigus)."""
    if not answer or not extracts:
        return []
    cleaned = _LIST_NUMBER.sub(" ", _CITATION_ZONE.sub(" ", answer))
    present = numbers_of(extracts)
    outside: list[str] = []
    for m in _NUMBER.finditer(cleaned):
        integer = _THOUSANDS.sub("", m.group(1))
        key = _key(m.group(1), m.group(2))
        if "." not in key and len(integer) < min_digits:
            continue
        # « 2 830 » écrit « 2830 » ou « 2.830 » dans l'extrait, et l'inverse
        if key in present or integer in present or key.replace(".", "") in present:
            continue
        raw = m.group(0).strip()
        if raw not in outside:
            outside.append(raw)
    return outside


# --- bilan ------------------------------------------------------------------
@dataclass
class GuardReport:
    evasive: bool = False
    degenerate: bool = False
    truncated: bool = False
    pages_outside: list[str] = field(default_factory=list)
    numbers_outside: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (
            self.evasive
            or self.degenerate
            or self.truncated
            or self.pages_outside
            or self.numbers_outside
        )

    def to_dict(self) -> dict:
        return {
            "evasive": self.evasive,
            "degenerate": self.degenerate,
            "truncated": self.truncated,
            "pages_outside": list(self.pages_outside),
            "numbers_outside": list(self.numbers_outside),
        }
