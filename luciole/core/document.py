"""Modèle documentaire commun : ce que tout parser produit.

V5 ramenait chaque fichier à une seule chaîne de texte et quelques
métadonnées globales ; le découpeur devait ensuite redeviner les tableaux,
les titres et les pages (voir ``docs/cadrage/analyse-v5.md``, point 5.2).
Ici un parser produit un ``ParsedDocument`` : une liste ordonnée de
``DocumentNode`` (titre, paragraphe, tableau, tour de parole…) qui gardent
la page, le chemin de section, le parent, la position et la méthode
d'extraction. Le découpage (étape 2), l'indexation et les citations
travaillent sur ces nœuds et n'ont rien à redécouvrir.

Les projections ``to_markdown()`` et ``to_text()`` sont dérivées des nœuds,
jamais l'inverse.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from luciole.core.versions import PARSER_VERSION

NodeType = Literal[
    "title",  # titre du document
    "heading",  # titre de section (``level`` donne la profondeur)
    "paragraph",
    "list_item",
    "table",  # un tableau entier, texte en Markdown
    "caption",
    "header",  # en-tête de page répété
    "footer",  # pied de page répété
    "figure_text",  # texte lu dans une figure, une carte, un schéma
    "code",
    "speaker_turn",  # tour de parole d'une transcription (``attributes.speaker``)
    "page_break",
]

ExtractionMethod = Literal[
    "native_text",  # texte brut de la page (PyMuPDF get_text)
    "markdown",  # mise en forme reconstruite (pymupdf4llm)
    "ocr",  # reconnaissance optique
    "docx",
    "plain",  # fichier texte ou Markdown lu tel quel
    "transcript",
]

# Types de nœuds qui portent une position dans la hiérarchie des sections.
_HEADING_TYPES = {"title", "heading"}


class BBox(BaseModel):
    """Rectangle en points PDF (origine en haut à gauche), ou en unités du format source."""

    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def _ordered(self) -> BBox:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("bbox : x1 >= x0 et y1 >= y0 attendus")
        return self


class DocumentNode(BaseModel):
    """Un bloc du document, dans l'ordre de lecture."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    parent_id: str | None = None
    node_type: NodeType
    text: str
    reading_order: int = Field(ge=0)
    page_start: int | None = Field(default=None, ge=1, description="Page en base 1")
    page_end: int | None = Field(default=None, ge=1)
    section_path: tuple[str, ...] = Field(
        default=(), description="Titres des sections englobantes, du plus haut au plus proche"
    )
    level: int | None = Field(default=None, ge=1, description="Profondeur d'un titre")
    bbox: BBox | None = None
    method: ExtractionMethod | None = None
    confidence: float | None = Field(default=None, ge=0, le=1, description="Confiance OCR")
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _coherent(self) -> DocumentNode:
        if self.page_start is not None and self.page_end is None:
            object.__setattr__(self, "page_end", self.page_start)
        if self.page_end is not None and self.page_start is None:
            raise ValueError("page_end sans page_start")
        if self.page_start is not None and self.page_end < self.page_start:
            raise ValueError("page_end < page_start")
        if self.node_type in _HEADING_TYPES and self.level is None:
            object.__setattr__(self, "level", 1 if self.node_type == "title" else 2)
        return self


class PageInfo(BaseModel):
    """Ce que l'on sait d'une page : méthode retenue, qualité, replis."""

    model_config = ConfigDict(extra="forbid")

    number: int = Field(ge=1)
    method: ExtractionMethod | None = Field(
        default=None, description="None : aucune méthode n'a produit de texte"
    )
    quality: float = Field(default=0.0, ge=0, le=1)
    char_count: int = Field(default=0, ge=0)
    needs_ocr: bool = False
    ocr_confidence: float | None = Field(default=None, ge=0, le=1)
    fallbacks: list[str] = Field(
        default_factory=list, description="Méthodes essayées puis écartées, dans l'ordre"
    )
    notes: list[str] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    """Résultat d'un parser : nœuds, pages et métadonnées du fichier."""

    model_config = ConfigDict(extra="forbid")

    file_name: str
    source_path: str = Field(description="Chemin tel que reçu par le parser")
    doc_type: str = Field(description="pdf, docx, md, txt, transcript…")
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    pages: list[PageInfo] = Field(default_factory=list)
    nodes: list[DocumentNode] = Field(default_factory=list)
    parser_version: int = PARSER_VERSION

    @model_validator(mode="after")
    def _graph_is_consistent(self) -> ParsedDocument:
        ids = [n.node_id for n in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("node_id en double")
        known = set(ids)
        orders = [n.reading_order for n in self.nodes]
        if orders != sorted(orders) or len(orders) != len(set(orders)):
            raise ValueError("reading_order doit être strictement croissant")
        for n in self.nodes:
            if n.parent_id is not None and n.parent_id not in known:
                raise ValueError(f"nœud {n.node_id} : parent {n.parent_id} inconnu")
        page_numbers = [p.number for p in self.pages]
        if page_numbers != sorted(page_numbers) or len(page_numbers) != len(set(page_numbers)):
            raise ValueError("pages : numéros croissants et uniques attendus")
        return self

    # ------------------------------------------------------------------
    # Accès
    # ------------------------------------------------------------------
    @property
    def page_count(self) -> int:
        return len(self.pages)

    def nodes_on_page(self, number: int) -> list[DocumentNode]:
        return [
            n
            for n in self.nodes
            if n.page_start is not None and n.page_start <= number <= n.page_end
        ]

    def children_of(self, node_id: str | None) -> list[DocumentNode]:
        return [n for n in self.nodes if n.parent_id == node_id]

    def headings(self) -> list[DocumentNode]:
        return [n for n in self.nodes if n.node_type in _HEADING_TYPES]

    # ------------------------------------------------------------------
    # Projections
    # ------------------------------------------------------------------
    def to_text(self, types: Iterable[NodeType] | None = None) -> str:
        """Texte brut, un nœud par bloc, sans en-têtes ni pieds répétés
        (destiné à BM25 et aux comparaisons de rendement)."""
        exclude = {"header", "footer", "page_break"}
        wanted = set(types) if types is not None else None
        parts = [
            n.text
            for n in self.nodes
            if n.node_type not in exclude and (wanted is None or n.node_type in wanted) and n.text
        ]
        return "\n\n".join(parts)

    def to_markdown(self) -> str:
        """Markdown lisible dérivé des nœuds (affichage, relecture)."""
        out: list[str] = []
        for n in self.nodes:
            if n.node_type in ("header", "footer", "page_break") or not n.text:
                continue
            if n.node_type == "title":
                out.append(f"# {n.text}")
            elif n.node_type == "heading":
                out.append(f"{'#' * min(n.level or 2, 6)} {n.text}")
            elif n.node_type == "list_item":
                out.append(f"- {n.text}")
            elif n.node_type == "code":
                out.append(f"```\n{n.text}\n```")
            elif n.node_type == "speaker_turn":
                speaker = n.attributes.get("speaker", "")
                stamp = n.attributes.get("timestamp")
                prefix = f"[{stamp}] " if stamp else ""
                out.append(f"**{prefix}{speaker} :** {n.text}" if speaker else n.text)
            elif n.node_type == "caption":
                out.append(f"_{n.text}_")
            else:
                out.append(n.text)
        return "\n\n".join(out)


class NodeBuilder:
    """Construit les nœuds dans l'ordre de lecture en tenant la hiérarchie.

    Le parser appelle ``add()`` bloc après bloc ; le builder attribue les
    identifiants, le ``reading_order``, le ``parent_id`` (le dernier titre de
    niveau supérieur) et le ``section_path``. Les parsers n'ont donc pas à
    gérer la pile des titres eux-mêmes.
    """

    def __init__(self, prefix: str = "n") -> None:
        self._prefix = prefix
        self._nodes: list[DocumentNode] = []
        # Pile de (level, node) des titres ouverts, du plus haut au plus profond.
        self._stack: list[tuple[int, DocumentNode]] = []

    def add(
        self,
        node_type: NodeType,
        text: str,
        *,
        page: int | None = None,
        page_end: int | None = None,
        level: int | None = None,
        bbox: BBox | None = None,
        method: ExtractionMethod | None = None,
        confidence: float | None = None,
        attributes: dict | None = None,
    ) -> DocumentNode:
        order = len(self._nodes)
        if node_type in _HEADING_TYPES:
            lvl = level if level is not None else (1 if node_type == "title" else 2)
            while self._stack and self._stack[-1][0] >= lvl:
                self._stack.pop()
        else:
            lvl = level
        parent = self._stack[-1][1] if self._stack else None
        node = DocumentNode(
            node_id=f"{self._prefix}{order:05d}",
            parent_id=parent.node_id if parent else None,
            node_type=node_type,
            text=text,
            reading_order=order,
            page_start=page,
            page_end=page_end,
            section_path=tuple(h.text for _, h in self._stack),
            level=lvl,
            bbox=bbox,
            method=method,
            confidence=confidence,
            attributes=attributes or {},
        )
        self._nodes.append(node)
        if node_type in _HEADING_TYPES:
            self._stack.append((lvl, node))
        return node

    def nodes(self) -> list[DocumentNode]:
        return list(self._nodes)
