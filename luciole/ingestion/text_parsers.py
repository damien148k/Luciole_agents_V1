"""Parsers sans dépendance externe : texte brut, Markdown, transcriptions.

Ils servent de référence au modèle documentaire et couvrent le cas d'usage
« commerciaux » (comptes rendus et transcriptions de réunions), dont les
documents arrivent le plus souvent en texte ou en Markdown.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from luciole.core.document import ExtractionMethod, NodeBuilder, PageInfo, ParsedDocument
from luciole.ingestion.base import Parser

_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*\S)\s*$")
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_FENCE_RE = re.compile(r"^\s*```")

# Tour de parole : « [00:12:03] Nom : texte », « (12:03) Nom : texte » ou
# « Nom : texte ». Le nom fait au plus quatre mots, commence par une
# majuscule, et le séparateur est « : » ou « - » entouré d'espaces.
_TURN_RE = re.compile(
    r"^\s*(?:[\[(](?P<stamp>\d{1,2}:\d{2}(?::\d{2})?)[\])]\s*)?"
    r"(?P<speaker>[A-ZÀ-Ý][\w'’.-]*(?:\s+[\wÀ-ÿ'’.-]+){0,3})\s*(?::|-)\s+(?P<text>.+\S)\s*$"
)


def _read(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path} : encodage illisible (ni UTF-8 ni cp1252)")


def _split_front_matter(text: str) -> tuple[dict, str]:
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return {}, text
    body = text[m.end() :]
    # Un bloc délimité par --- est du front matter même s'il est illisible :
    # on le retire du texte et on garde des métadonnées vides.
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return {}, body
    if not isinstance(meta, dict):
        return {}, body
    flat = {
        k: (v if isinstance(v, str | int | float | bool) or v is None else str(v))
        for k, v in meta.items()
    }
    return flat, body


def _single_page(char_count: int, method: str) -> list[PageInfo]:
    """Un fichier texte est une page unique de qualité pleine s'il a du contenu."""
    return [
        PageInfo(
            number=1,
            method=method if char_count else None,
            quality=1.0 if char_count else 0.0,
            char_count=char_count,
        )
    ]


class TextParser(Parser):
    """Texte brut : un paragraphe par bloc séparé d'une ligne vide."""

    extensions = (".txt",)

    def parse(self, path: str | Path) -> ParsedDocument:
        path = Path(path)
        text = _read(path)
        b = NodeBuilder()
        for block in re.split(r"\n\s*\n", text):
            block = block.strip()
            if block:
                b.add("paragraph", block, page=1, method="plain")
        return ParsedDocument(
            file_name=path.name,
            source_path=str(path),
            doc_type="txt",
            pages=_single_page(len(text.strip()), "plain"),
            nodes=b.nodes(),
        )


class MarkdownParser(Parser):
    """Markdown : titres, listes, tableaux, blocs de code, front matter YAML."""

    extensions = (".md", ".markdown")

    def parse(self, path: str | Path) -> ParsedDocument:
        path = Path(path)
        meta, body = _split_front_matter(_read(path))
        b = NodeBuilder()
        add_markdown_blocks(b, body)
        return ParsedDocument(
            file_name=path.name,
            source_path=str(path),
            doc_type="md",
            metadata=meta,
            pages=_single_page(len(body.strip()), "plain"),
            nodes=b.nodes(),
        )


def add_markdown_blocks(
    b: NodeBuilder, body: str, *, page: int = 1, method: ExtractionMethod = "plain"
) -> None:
    """Ajoute à ``b`` les blocs d'un texte Markdown (titres, listes, tableaux,
    code, paragraphes). Partagé par le parser Markdown et le parser PDF, qui
    reçoit du Markdown de pymupdf4llm page par page."""
    lines = body.splitlines()
    i = 0
    para: list[str] = []
    first_heading_seen = False

    def flush_para() -> None:
        if para:
            b.add("paragraph", " ".join(s.strip() for s in para), page=page, method=method)
            para.clear()

    while i < len(lines):
        line = lines[i]
        if _FENCE_RE.match(line):
            flush_para()
            j = i + 1
            while j < len(lines) and not _FENCE_RE.match(lines[j]):
                j += 1
            b.add("code", "\n".join(lines[i + 1 : j]), page=page, method=method)
            i = j + 1
            continue
        m = _HEADING_RE.match(line)
        if m:
            flush_para()
            level = len(m.group(1))
            # Le premier titre de niveau 1 est le titre du document.
            if level == 1 and not first_heading_seen:
                b.add("title", m.group(2), page=page, level=1, method=method)
            else:
                b.add("heading", m.group(2), page=page, level=level, method=method)
            first_heading_seen = True
            i += 1
            continue
        if _TABLE_LINE_RE.match(line):
            flush_para()
            j = i
            while j < len(lines) and _TABLE_LINE_RE.match(lines[j]):
                j += 1
            rows = lines[i:j]
            b.add(
                "table",
                "\n".join(r.strip() for r in rows),
                page=page,
                method=method,
                attributes={"rows": len(rows)},
            )
            i = j
            continue
        m = _LIST_RE.match(line)
        if m:
            flush_para()
            b.add("list_item", m.group(1), page=page, method=method)
            i += 1
            continue
        if not line.strip():
            flush_para()
        else:
            para.append(line)
        i += 1
    flush_para()


class TranscriptParser(Parser):
    """Transcription ou compte rendu de réunion : un nœud par tour de parole.

    Reconnaît ``[hh:mm:ss] Nom : texte`` et ``Nom : texte``. Les lignes qui ne
    sont pas un tour de parole prolongent le tour précédent (texte sur
    plusieurs lignes) ou forment un paragraphe. Les titres Markdown sont
    gardés (ex. ``## Réunion 2 — 4 mars 2026``) et le front matter YAML
    alimente les métadonnées (client, produit, commercial, numéro de réunion…).
    """

    extensions = (".transcript.md", ".vtt.txt", ".transcript.txt")

    def parse(self, path: str | Path) -> ParsedDocument:
        path = Path(path)
        meta, body = _split_front_matter(_read(path))
        b = NodeBuilder()
        current: dict | None = None
        speakers: set[str] = set()

        def flush() -> None:
            nonlocal current
            if current:
                b.add(
                    "speaker_turn",
                    " ".join(current["lines"]),
                    page=1,
                    method="transcript",
                    attributes={
                        k: v
                        for k, v in (
                            ("speaker", current["speaker"]),
                            ("timestamp", current["stamp"]),
                        )
                        if v
                    },
                )
                current = None

        for raw in body.splitlines():
            line = raw.rstrip()
            if not line.strip():
                flush()
                continue
            h = _HEADING_RE.match(line)
            if h:
                flush()
                b.add("heading", h.group(2), page=1, level=len(h.group(1)), method="transcript")
                continue
            t = _TURN_RE.match(line)
            if t:
                flush()
                speaker = t.group("speaker").strip()
                speakers.add(speaker)
                current = {
                    "speaker": speaker,
                    "stamp": t.group("stamp"),
                    "lines": [t.group("text")],
                }
                continue
            if current:
                current["lines"].append(line.strip())
            else:
                b.add("paragraph", line.strip(), page=1, method="transcript")
        flush()

        meta = dict(meta)
        meta.setdefault("speaker_count", len(speakers))
        return ParsedDocument(
            file_name=path.name,
            source_path=str(path),
            doc_type="transcript",
            metadata=meta,
            pages=_single_page(len(body.strip()), "transcript"),
            nodes=b.nodes(),
        )


def default_text_registry():
    """Registre des parsers sans dépendance externe. Les suffixes composés
    (``.transcript.md``) sont testés avant les simples."""

    return _CompoundSuffixRegistry([TranscriptParser(), MarkdownParser(), TextParser()])


class _CompoundSuffixRegistry:
    """Comme ``ParserRegistry`` mais accepte des suffixes composés
    (``.transcript.md``), testés du plus long au plus court."""

    def __init__(self, parsers: list[Parser]) -> None:
        self._by_ext: dict[str, Parser] = {}
        for p in parsers:
            for ext in p.extensions:
                if ext in self._by_ext:
                    raise ValueError(f"extension {ext} déjà enregistrée")
                self._by_ext[ext.lower()] = p

    def supported_extensions(self) -> list[str]:
        return sorted(self._by_ext)

    def _match(self, path: Path) -> str | None:
        name = path.name.lower()
        for ext in sorted(self._by_ext, key=len, reverse=True):
            if name.endswith(ext) and len(name) > len(ext):
                return ext
        return None

    def supports(self, path: str | Path) -> bool:
        return self._match(Path(path)) is not None

    def parser_for(self, path: str | Path) -> Parser:
        ext = self._match(Path(path))
        if ext is None:
            raise ValueError(
                f"format non pris en charge : {Path(path).suffix or '(sans extension)'} ; "
                f"formats connus : {', '.join(self.supported_extensions())}"
            )
        return self._by_ext[ext]

    def parse(self, path: str | Path) -> ParsedDocument:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return self.parser_for(path).parse(path)
