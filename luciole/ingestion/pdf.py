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
    ParsedDocument,
)
from luciole.ingestion.base import Parser
from luciole.ingestion.ocr import OcrEngine, group_lines, render_page, to_points
from luciole.ingestion.text_parsers import add_markdown_blocks

_DIGITS_RE = re.compile(r"\d+")
_SPACES_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w#]+")


def _normalize(line: str) -> str:
    """Forme comparable d'une ligne d'en-tête ou de pied : chiffres neutralisés
    (« Page 3 / 120 » et « Page 4 / 120 » se correspondent), espaces réduits."""
    neutral = _PUNCT_RE.sub(" ", _DIGITS_RE.sub("#", line.lower()))
    return _SPACES_RE.sub(" ", neutral).strip()


@dataclass
class _Block:
    """Un bloc en attente d'ajout, pour trier une page par position."""

    node_type: NodeType
    text: str
    method: ExtractionMethod
    bbox: BBox | None = None
    confidence: float | None = None
    attributes: dict = field(default_factory=dict)

    def sort_key(self) -> tuple[float, float]:
        if self.bbox is None:
            return (math.inf, 0.0)
        return (round(self.bbox.y0 / 4) * 4, self.bbox.x0)


class PdfParser(Parser):
    extensions = (".pdf",)

    def __init__(
        self,
        *,
        ocr: OcrEngine | None = None,
        min_chars_per_page: int = 50,
        min_words_per_page: int = 8,
        max_drawings_per_page: int = 500,
        min_yield_ratio: float = 0.5,
        yield_reference_chars: int = 200,
        ocr_dpi: int = 300,
        repeated_line_ratio: float = 0.4,
    ) -> None:
        self.ocr = ocr
        self.min_chars_per_page = min_chars_per_page
        self.min_words_per_page = min_words_per_page
        self.max_drawings_per_page = max_drawings_per_page
        self.min_yield_ratio = min_yield_ratio
        self.yield_reference_chars = yield_reference_chars
        self.ocr_dpi = ocr_dpi
        self.repeated_line_ratio = repeated_line_ratio

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
        builder = NodeBuilder()
        pages: list[PageInfo] = []
        methods: Counter[str] = Counter()

        for index, page in enumerate(doc):
            number = index + 1
            native = natives[index]
            info, blocks = self._parse_page(doc, page, number, native, repeated)
            pages.append(info)
            methods[info.method or "none"] += 1
            for blk in sorted(blocks, key=_Block.sort_key):
                builder.add(
                    blk.node_type,
                    blk.text,
                    page=number,
                    method=blk.method,
                    bbox=blk.bbox,
                    confidence=blk.confidence,
                    attributes=blk.attributes,
                )

        meta = doc.metadata or {}
        needs_ocr = sum(1 for p in pages if p.needs_ocr)
        logger.info(
            f"{path.name} : {len(pages)} pages, méthodes {dict(methods)}, {needs_ocr} page(s) à OCR"
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
                "pages_need_ocr": needs_ocr,
            },
            pages=pages,
            nodes=builder.nodes(),
        )

    # ------------------------------------------------------------------
    def _repeated_lines(self, doc: pymupdf.Document) -> dict[str, NodeType]:
        """Bloc le plus haut et bloc le plus bas de chaque page ; ceux qui
        reviennent sur assez de pages sont des en-têtes (haut) ou des pieds
        (bas). Les chiffres sont neutralisés pour que « Page 3 / 120 » et
        « Page 4 / 120 » se correspondent."""
        if len(doc) < 3:
            return {}
        counts: Counter[tuple[str, NodeType]] = Counter()
        for page in doc:
            blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
            if not blocks:
                continue
            blocks.sort(key=lambda b: b[1])
            top = _normalize(blocks[0][4].splitlines()[0])
            bottom = _normalize(blocks[-1][4].strip().splitlines()[-1])
            seen = {(top, "header"), (bottom, "footer")}
            counts.update(seen)
        threshold = max(3, math.ceil(self.repeated_line_ratio * len(doc)))
        return {
            line: kind for (line, kind), n in counts.items() if n >= threshold and len(line) >= 3
        }

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
        return chunks[0].get("text") or ""

    def _native_blocks(self, page: pymupdf.Page, repeated: dict[str, NodeType]) -> list[_Block]:
        out: list[_Block] = []
        for x0, y0, x1, y1, text, _no, btype in page.get_text("blocks"):
            if btype != 0:  # 1 = image
                continue
            text = " ".join(text.split())
            if not text:
                continue
            node_type: NodeType = repeated.get(_normalize(text), "paragraph")
            out.append(
                _Block(
                    node_type=node_type,
                    text=text,
                    method="native_text",
                    bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
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
