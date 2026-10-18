"""Morceau (chunk) : l'unité que l'on indexe, retrouve et cite.

Deux niveaux, liés par ``parent_id`` :

- **parent** : une section du document (ou un tronçon d'une longue section),
  ce que l'on donne à lire au LLM autour d'un passage retrouvé ;
- **enfant** : un passage de quelques centaines de tokens à l'intérieur d'un
  parent, ce que l'on cherche (embedding, BM25) et que l'on cite.

Chaque morceau porte de quoi être tracé sans relire le fichier (règle 4 du
projet) : document, dossier, hash du contenu, pages PDF et numéros
imprimés, chemin de section, nœuds d'origine, méthodes d'extraction,
versions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ChunkKind = Literal["parent", "child"]


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    kind: ChunkKind
    parent_id: str | None = Field(default=None, description="Parent d'un enfant ; None sinon")
    index: int = Field(ge=0, description="Rang parmi les morceaux de même niveau du document")

    document_id: str
    dossier_id: str
    relative_path: str
    file_name: str
    doc_type: str
    content_hash: str

    text: str = Field(min_length=1, description="Texte du passage")
    context: str = Field("", description="Titre du document et chemin de section")
    token_count: int = Field(ge=1, description="Tokens de embed_text, spéciaux compris")

    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    page_labels: list[str] = Field(
        default_factory=list, description="Numéros imprimés couverts (« 7/47 »), dans l'ordre"
    )
    section_path: tuple[str, ...] = ()
    node_ids: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list, description="Méthodes d'extraction")
    metadata: dict[str, object] = Field(
        default_factory=dict, description="Métadonnées validées du document"
    )
    versions: dict[str, str | int] = Field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        """Ce que voit le modèle d'embedding : contexte puis passage."""
        return f"{self.context}\n\n{self.text}" if self.context else self.text

    def citation(self) -> str:
        """Référence lisible : chemin dans la racine (le dossier lève l'ambiguïté
        entre deux fichiers de même nom), page imprimée si connue, page PDF."""
        if self.page_start is None:
            return self.relative_path
        pdf = (
            f"p. {self.page_start}"
            if self.page_start == self.page_end
            else f"p. {self.page_start}-{self.page_end}"
        )
        if not self.page_labels:
            return f"{self.relative_path}, {pdf}"
        printed = ", ".join(self.page_labels)
        return f"{self.relative_path}, page {printed} (PDF {pdf})"
