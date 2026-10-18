"""Parser PDF : routage et qualité **page par page**.

V5 décidait pour tout le document (moyenne de dessins par page, OCR étendu
sur toutes les pages si le total était court) et collait le texte OCR en
fin de document (``docs/cadrage/analyse-v5.md``, points 5.3 à 5.5). Ici
chaque page est inspectée seule et reçoit sa méthode :

- peu ou pas de texte natif → **OCR** si un moteur est fourni, sinon la page
  est marquée ``needs_ocr`` et ce qui existe en natif est gardé ;
- page vectorielle complexe (trop de tracés : plan, carte) → **texte natif**
  par blocs, sans passer par pymupdf4llm qui peut y bloquer ;
- sinon → **Markdown** pymupdf4llm (titres, tableaux), avec le garde-fou de
  rendement de V5 : si la page perd trop de texte par rapport au natif,
  repli sur le texte natif de cette page seulement.

Les en-têtes et pieds de page répétés sont reconnus et typés ``header`` /
``footer`` : ils sortent des projections et des futurs embeddings. Les
nœuds OCR gardent boîte et confiance. Tout est consigné dans ``PageInfo``
(méthode retenue, qualité, replis, notes).

Réglages mesurés sur de vraies études d'impact (A3 paysage, 202 pages) :

- les coordonnées des blocs sont ramenées dans le repère de la page
  **affichée** (``rotation_matrix``) : un PDF portrait tourné de 90° se lit
  sinon dans le mauvais ordre et ses en-têtes ne sont pas reconnus ;
- une page sans texte, sans image et sans tracé est **vide** (verso d'une
  double page), pas une page à OCR ;
- les pages A3 « deux pages côte à côte » sont lues colonne par colonne ;
- le seuil « vectoriel complexe » est à 5000 tracés : jusqu'à ~2000 tracés
  ce sont des tableaux que pymupdf4llm rend très bien (< 3 s) ; au-delà de
  7000 ce sont des cartes, sans tableau, où le texte natif suffit ;
- les numéros de page **imprimés** (« 7/47 » en pied de chaque moitié d'une
  double page, « 12 » en début de pied d'un tome) sont relevés dans les
  bandes de marge et rangés dans ``PageInfo.labels`` pour les citations.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
import pymupdf4llm
from loguru import logger

from luciole.core.document import (
    BBox,
    DocumentNode,
    ExtractionMethod,
    NodeBuilder,
    NodeType,
    PageInfo,
    PageLabel,
    ParsedDocument,
)
from luciole.ingestion.base import Parser
from luciole.ingestion.ocr import OcrEngine, group_lines, render_page, to_points
from luciole.ingestion.text_parsers import add_markdown_blocks

_DIGITS_RE = re.compile(r"\d+")
_SPACES_RE = re.compile(r"\s+")
_DASHES_RE = re.compile(r"[‐-―−]")
_MARKUP_RE = re.compile(r"[*`_]+")
_LETTER_RE = re.compile(r"[^\W\d_]")
_PAGE_NUMBER_RE = re.compile(r"^(page|p\.)?\s*#\s*([/-]|sur|of)?\s*#?$")
# Numéro imprimé seul sur sa ligne de marge : « 7/47 », « Page 3 sur 12 », « 12 ».
# Pas de « / » juste avant : « 01/06/2021 » n'est pas la page 6 sur 2021.
_LABEL_FRACTION_RE = re.compile(
    r"(?:^|(?<=[\s|]))(?:page\s*|p\.\s*)?(\d{1,4})\s*(?:/|sur|of)\s*(\d{1,4})$", re.IGNORECASE
)
_LABEL_BARE_RE = re.compile(r"^(?:page\s*|p\.\s*)?(\d{1,4})$", re.IGNORECASE)
_BLANK_NOTE = "page vide : ni texte, ni image, ni tracé"
_MAX_MARGIN_LINE = 160  # au-delà, une ligne de bande n'est plus un en-tête mais du texte


def _normalize(line: str) -> str:
    """Forme comparable d'une ligne d'en-tête ou de pied : chiffres neutralisés
    (« Page 3 / 120 » et « Page 4 / 120 » se correspondent), tirets unifiés,
    balises Markdown (``**``) et espaces réduits. La ponctuation est gardée :
    « 14/47 » (numéro de page) et « 0,0 » (cellule de tableau) restent distincts."""
    neutral = _DIGITS_RE.sub("#", _MARKUP_RE.sub("", _DASHES_RE.sub("-", line.lower())))
    return _SPACES_RE.sub(" ", neutral).strip()


_PICTURE_TEXT_RE = re.compile(r"<!--\s*(Start|End) of picture text\s*-->")
_HTML_TAGS_RE = re.compile(r"</?(mark|b|i|u|sup|sub)>")
_BR_RE = re.compile(r"<br\s*/?>")


def _clean_markdown(md: str) -> str:
    """Retire les balises HTML que pymupdf4llm glisse dans son Markdown (vu sur
    de vraies études : ``<!-- Start of picture text -->``, ``<br>``, ``<mark>``)."""
    md = _HTML_TAGS_RE.sub("", _PICTURE_TEXT_RE.sub("", md))
    # Un <br> dans une ligne de tableau casserait la ligne : il devient une espace.
    lines = [
        _BR_RE.sub(" ", line) if line.lstrip().startswith("|") else _BR_RE.sub("\n", line)
        for line in md.splitlines()
    ]
    return "\n".join(lines)


def _is_margin_line(normalized: str) -> bool:
    """Une ligne de bande est un en-tête ou un pied plausible si elle porte des
    lettres, ou si c'est un numéro de page ; « 0,0 » ou « [135° ; 315°[ »
    (cellules de tableau en bas de page) n'en sont pas."""
    if not 3 <= len(normalized) <= _MAX_MARGIN_LINE:
        return normalized == "#"
    return bool(_LETTER_RE.search(normalized)) or bool(_PAGE_NUMBER_RE.match(normalized))


@dataclass
class _Block:
    """Un bloc en attente d'ajout, pour trier une page par position."""

    node_type: NodeType
    text: str
    method: ExtractionMethod
    bbox: BBox | None = None
    confidence: float | None = None
    attributes: dict = field(default_factory=dict)
    column: int = 0
    level: int | None = None

    def sort_key(self) -> tuple[int, float, float]:
        if self.bbox is None:
            return (self.column, math.inf, 0.0)
        return (self.column, round(self.bbox.y0 / 4) * 4, self.bbox.x0)


def _text_blocks(page: pymupdf.Page) -> list[tuple[pymupdf.Rect, str]]:
    """Blocs de texte natif avec leur boîte dans le repère de la page affichée
    (``get_text`` rend des coordonnées non tournées)."""
    out = []
    for x0, y0, x1, y1, text, _no, btype in page.get_text("blocks"):
        if btype != 0 or not text.strip():  # 1 = image
            continue
        rect = pymupdf.Rect(x0, y0, x1, y1) * page.rotation_matrix
        rect.normalize()
        out.append((rect, text))
    return out


class PdfParser(Parser):
    extensions = (".pdf",)

    def __init__(
        self,
        *,
        ocr: OcrEngine | None = None,
        min_chars_per_page: int = 50,
        min_words_per_page: int = 8,
        max_drawings_per_page: int = 5000,
        min_yield_ratio: float = 0.5,
        yield_reference_chars: int = 200,
        ocr_dpi: int = 300,
        repeated_line_ratio: float = 0.1,
        repeated_line_min_pages: int = 3,
        margin_band_ratio: float = 0.12,
    ) -> None:
        self.ocr = ocr
        self.min_chars_per_page = min_chars_per_page
        self.min_words_per_page = min_words_per_page
        self.max_drawings_per_page = max_drawings_per_page
        self.min_yield_ratio = min_yield_ratio
        self.yield_reference_chars = yield_reference_chars
        self.ocr_dpi = ocr_dpi
        # Une ligne de la bande haute (ou basse) de la page est un en-tête (ou
        # un pied) si elle revient sur max(min_pages, ratio × pages) pages.
        self.repeated_line_ratio = repeated_line_ratio
        self.repeated_line_min_pages = repeated_line_min_pages
        self.margin_band_ratio = margin_band_ratio

    # ------------------------------------------------------------------
    def parse(self, path: str | Path) -> ParsedDocument:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        doc = pymupdf.open(path)
        try:
            return self._parse_doc(doc, path)
        finally:
            doc.close()

    def _parse_doc(self, doc: pymupdf.Document, path: Path) -> ParsedDocument:
        natives = [page.get_text("text") for page in doc]
        repeated = self._repeated_lines(doc)
        printed = self._printed_labels(doc)
        builder = NodeBuilder()
        pages: list[PageInfo] = []
        methods: Counter[str] = Counter()
        has_title = False

        for index, page in enumerate(doc):
            number = index + 1
            native = natives[index]
            info, blocks = self._parse_page(doc, page, number, native, repeated)
            info.width, info.height = page.rect.width, page.rect.height
            info.labels = printed.get(number, [])
            if len(info.labels) == 2:
                self._mark_sides(page, blocks)
            pages.append(info)
            methods[info.method or "none"] += 1
            for blk in sorted(blocks, key=_Block.sort_key):
                kind = blk.node_type
                if kind == "title":
                    # pymupdf4llm est appelé page par page : seul le premier
                    # titre de niveau 1 du document en est le titre.
                    kind = "heading" if has_title else "title"
                    has_title = True
                builder.add(
                    kind,
                    blk.text,
                    page=number,
                    level=blk.level,
                    method=blk.method,
                    bbox=blk.bbox,
                    confidence=blk.confidence,
                    attributes=blk.attributes,
                )

        meta = doc.metadata or {}
        needs_ocr = sum(1 for p in pages if p.needs_ocr)
        blank = sum(1 for p in pages if _BLANK_NOTE in p.notes)
        logger.info(
            f"{path.name} : {len(pages)} pages, méthodes {dict(methods)}, "
            f"{needs_ocr} page(s) à OCR, {blank} page(s) vide(s)"
        )
        return ParsedDocument(
            file_name=path.name,
            source_path=str(path),
            doc_type="pdf",
            metadata={
                "title": meta.get("title") or None,
                "author": meta.get("author") or None,
                "creation_date": meta.get("creationDate") or None,
                "page_count": len(pages),
                "pages_markdown": methods.get("markdown", 0),
                "pages_native_text": methods.get("native_text", 0),
                "pages_ocr": methods.get("ocr", 0),
                "pages_without_text": methods.get("none", 0),
                "pages_blank": blank,
                "pages_need_ocr": needs_ocr,
                "pages_with_printed_label": len(printed),
            },
            pages=pages,
            nodes=builder.nodes(),
        )

    # ------------------------------------------------------------------
    def _margin_bands(self, page: pymupdf.Page) -> tuple[float, float]:
        """Limites (y) de la bande haute et de la bande basse de la page affichée."""
        h = page.rect.height
        return h * self.margin_band_ratio, h * (1 - self.margin_band_ratio)

    def _repeated_lines(self, doc: pymupdf.Document) -> dict[str, NodeType]:
        """Lignes de la bande haute et de la bande basse de chaque page ; celles
        qui reviennent sur assez de pages sont des en-têtes (haut) ou des pieds
        (bas). Les chiffres sont neutralisés pour que « Page 3 / 120 » et
        « Page 4 / 120 » se correspondent. On regarde toute la bande, pas un
        seul bloc : une double page A3 porte deux en-têtes et deux pieds, et
        un logo ou une légende peut descendre plus bas que le pied."""
        if len(doc) < self.repeated_line_min_pages:
            return {}
        counts: Counter[tuple[str, NodeType]] = Counter()
        for page in doc:
            top_limit, bottom_limit = self._margin_bands(page)
            seen: set[tuple[str, NodeType]] = set()
            for rect, text in _text_blocks(page):
                if rect.y1 <= top_limit:
                    kind: NodeType = "header"
                elif rect.y0 >= bottom_limit:
                    kind = "footer"
                else:
                    continue
                # Le bloc entier (forme des nœuds natifs) et chaque ligne
                # (forme des nœuds Markdown) sont comptés.
                for raw in (text, *text.splitlines()):
                    line = _normalize(raw)
                    if _is_margin_line(line):
                        seen.add((line, kind))
            counts.update(seen)
        threshold = max(
            self.repeated_line_min_pages, math.ceil(self.repeated_line_ratio * len(doc))
        )
        return {line: kind for (line, kind), n in counts.items() if n >= threshold}

    def _printed_labels(self, doc: pymupdf.Document) -> dict[int, list[PageLabel]]:
        """Numéros de page imprimés, par numéro de page PDF.

        Une ligne de la bande haute ou basse qui n'est qu'un numéro (« 7/47 »,
        « 12 ») est candidate. Une forme (fraction ou nombre seul, en haut ou
        en bas) n'est retenue que si elle revient sur assez de pages (même
        seuil que les en-têtes) avec des valeurs qui changent : une cellule
        « 12 » isolée ou un « 2021 » répété ne sont pas des numéros de page.
        Au plus un numéro par moitié de page, le plus proche du bord."""
        if len(doc) < self.repeated_line_min_pages:
            return {}
        # (page, forme, bande) -> candidats (moitié, distance au bord, numéro, total)
        found: dict[tuple[int, str, str], list[tuple[str, float, str, int | None]]] = {}
        for page in doc:
            top_limit, bottom_limit = self._margin_bands(page)
            width, height = page.rect.width, page.rect.height
            for rect, text in _text_blocks(page):
                if rect.y1 <= top_limit:
                    band, edge = "top", rect.y0
                elif rect.y0 >= bottom_limit:
                    band, edge = "bottom", height - rect.y1
                else:
                    continue
                for raw in text.splitlines():
                    line = _MARKUP_RE.sub("", raw).strip()
                    if m := _LABEL_FRACTION_RE.search(line):
                        number, total = int(m.group(1)), int(m.group(2))
                        if not 1 <= number <= total:
                            continue
                        form = "fraction"
                    elif m := _LABEL_BARE_RE.match(line):
                        number, total, form = int(m.group(1)), None, "bare"
                        if number == 0:
                            continue
                    else:
                        continue
                    half = "left" if (rect.x0 + rect.x1) / 2 < width / 2 else "right"
                    found.setdefault((page.number + 1, form, band), []).append(
                        (half, edge, str(number), total)
                    )
        threshold = max(
            self.repeated_line_min_pages, math.ceil(self.repeated_line_ratio * len(doc))
        )
        accepted: set[tuple[str, str]] = set()
        for form, band in {(f, b) for _, f, b in found}:
            hits = [
                c for (_, f, b), cands in found.items() if (f, b) == (form, band) for c in cands
            ]
            pages = {p for (p, f, b) in found if (f, b) == (form, band)}
            if len(pages) >= threshold and len({c[2] for c in hits}) >= 0.8 * len(hits):
                accepted.add((form, band))
        out: dict[int, list[PageLabel]] = {}
        for (number, form, band), cands in sorted(found.items()):
            if (form, band) not in accepted or number in out:
                continue
            best: dict[str, tuple[float, str, int | None]] = {}
            for half, edge, label, total in cands:
                if half not in best or edge < best[half][0]:
                    best[half] = (edge, label, total)
            two = len(best) == 2
            out[number] = [
                PageLabel(label=label, total=total, side=half if two else "page")
                for half, (_, label, total) in sorted(best.items())
            ]
        return out

    @staticmethod
    def _mark_sides(page: pymupdf.Page, blocks: list[_Block]) -> None:
        """Double page : note de quelle moitié vient chaque bloc sans boîte (les
        blocs Markdown de pymupdf4llm n'en ont pas), en retrouvant son début
        dans un bloc de texte natif. Sert à citer « 7/47 » plutôt que
        « 7/47, 8/47 ». Un bloc introuvable (tableau recomposé…) reste sans côté."""
        middle = page.rect.width / 2
        natives = [
            (_SPACES_RE.sub(" ", _MARKUP_RE.sub("", text)).strip().lower(), rect)
            for rect, text in _text_blocks(page)
        ]
        for blk in blocks:
            if blk.bbox is not None:
                continue
            probe = _SPACES_RE.sub(" ", _MARKUP_RE.sub("", blk.text)).strip().lower()[:40]
            if len(probe) < 8:
                continue
            sides = {
                "left" if (rect.x0 + rect.x1) / 2 < middle else "right"
                for text, rect in natives
                if probe in text
            }
            if len(sides) == 1:
                blk.attributes["side"] = sides.pop()

    def _parse_page(
        self,
        doc: pymupdf.Document,
        page: pymupdf.Page,
        number: int,
        native: str,
        repeated: dict[str, NodeType],
    ) -> tuple[PageInfo, list[_Block]]:
        chars = len(native.strip())
        words = len(native.split())
        info = PageInfo(number=number, char_count=chars)
        blocks: list[_Block] = []

        drawings = len(page.get_drawings())
        if chars == 0 and drawings == 0 and not page.get_images():
            # Verso blanc d'une double page : rien à extraire, rien à OCR.
            info.quality = 0.0
            info.notes.append(_BLANK_NOTE)
            return info, blocks

        complex_vector = drawings > self.max_drawings_per_page
        if complex_vector:
            info.notes.append(f"page vectorielle complexe : {drawings} tracés")

        sparse = chars < self.min_chars_per_page or words < self.min_words_per_page
        if sparse:
            info.needs_ocr = True
            blocks += self._native_blocks(page, repeated)  # ce qui existe reste
            if self.ocr is None:
                info.method = "native_text" if blocks else None
                info.quality = 0.0
                info.notes.append("OCR requis : aucun moteur OCR disponible")
                return info, blocks
            ocr_blocks, confidence = self._ocr_blocks(page)
            blocks += ocr_blocks
            info.method = "ocr" if ocr_blocks else (info.method if blocks else None)
            info.ocr_confidence = confidence
            info.quality = confidence or 0.0
            if not ocr_blocks:
                info.notes.append("OCR sans résultat")
            return info, blocks

        if complex_vector:
            # pymupdf4llm peut bloquer très longtemps sur un plan : texte natif.
            info.method = "native_text"
            info.quality = 1.0
            return info, self._native_blocks(page, repeated)

        md = self._markdown(doc, page.number)
        if md is None:
            info.fallbacks.append("markdown")
            info.notes.append("pymupdf4llm en échec sur cette page")
            info.method = "native_text"
            info.quality = 1.0
            return info, self._native_blocks(page, repeated)

        md_chars = len(md.strip())
        ratio = md_chars / chars if chars else 1.0
        if chars >= self.yield_reference_chars and ratio < self.min_yield_ratio:
            info.fallbacks.append("markdown")
            info.notes.append(f"rendement markdown {ratio:.0%} < {self.min_yield_ratio:.0%}")
            info.method = "native_text"
            info.quality = 1.0
            return info, self._native_blocks(page, repeated)

        info.method = "markdown"
        info.quality = min(1.0, ratio)
        tmp = NodeBuilder()
        add_markdown_blocks(tmp, md, page=number, method="markdown")
        for n in tmp.nodes():
            blocks.append(
                _Block(
                    node_type=self._classify(n, repeated),
                    text=n.text,
                    method="markdown",
                    attributes=dict(n.attributes),
                    level=n.level if n.node_type in ("title", "heading") else None,
                )
            )
        return info, blocks

    # ------------------------------------------------------------------
    @staticmethod
    def _markdown(doc: pymupdf.Document, page_index: int) -> str | None:
        try:
            chunks = pymupdf4llm.to_markdown(doc, pages=[page_index], page_chunks=True)
        except Exception as e:  # noqa: BLE001 - bibliothèque tierce, on replie
            logger.warning(f"pymupdf4llm page {page_index + 1} : {e}")
            return None
        if not chunks:
            return None
        return _clean_markdown(chunks[0].get("text") or "")

    def _gutter(self, page: pymupdf.Page, blocks: list[tuple[pymupdf.Rect, str]]) -> float | None:
        """Abscisse de la gouttière d'une page paysage « deux pages côte à
        côte », ou ``None``. Il faut des blocs de corps des deux côtés et aucun
        qui chevauche le milieu (les en-têtes et pieds pleine largeur sont
        ignorés)."""
        if page.rect.width <= page.rect.height:
            return None
        center = page.rect.width / 2
        top_limit, bottom_limit = self._margin_bands(page)
        left = right = 0
        for rect, _ in blocks:
            if rect.y1 <= top_limit or rect.y0 >= bottom_limit:
                continue
            if rect.x0 < center < rect.x1:
                return None
            if rect.x1 <= center:
                left += 1
            else:
                right += 1
        return center if left and right else None

    def _native_blocks(self, page: pymupdf.Page, repeated: dict[str, NodeType]) -> list[_Block]:
        raw = _text_blocks(page)
        gutter = self._gutter(page, raw)
        out: list[_Block] = []
        for rect, text in raw:
            text = " ".join(text.split())
            node_type: NodeType = repeated.get(_normalize(text), "paragraph")
            out.append(
                _Block(
                    node_type=node_type,
                    text=text,
                    method="native_text",
                    bbox=BBox(x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1),
                    column=1 if gutter is not None and rect.x0 >= gutter else 0,
                )
            )
        return out

    def _ocr_blocks(self, page: pymupdf.Page) -> tuple[list[_Block], float | None]:
        assert self.ocr is not None
        rendered = render_page(page, self.ocr_dpi)
        lines = self.ocr.recognize(rendered.png)
        if not lines:
            return [], None
        blocks: list[_Block] = []
        for group in group_lines(lines):
            bboxes = [ln.bbox for ln in group]
            merged = BBox(
                x0=min(b.x0 for b in bboxes),
                y0=min(b.y0 for b in bboxes),
                x1=max(b.x1 for b in bboxes),
                y1=max(b.y1 for b in bboxes),
            )
            blocks.append(
                _Block(
                    node_type="paragraph",
                    text=" ".join(ln.text for ln in group),
                    method="ocr",
                    bbox=to_points(merged, rendered.scale),
                    confidence=sum(ln.confidence for ln in group) / len(group),
                    attributes={"ocr_engine": self.ocr.name},
                )
            )
        mean = sum(ln.confidence for ln in lines) / len(lines)
        return blocks, round(mean, 3)

    @staticmethod
    def _classify(node: DocumentNode, repeated: dict[str, NodeType]) -> NodeType:
        if node.node_type == "paragraph":
            return repeated.get(_normalize(node.text), "paragraph")
        return node.node_type
