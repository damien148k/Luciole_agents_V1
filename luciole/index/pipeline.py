"""Pipeline d'ingestion : une racine de dossiers → magasin + index alignés.

Un passage (``run``) :

1. parcourt la racine (``scan``), chaque fichier lu une seule fois ;
2. pour chaque document **nouveau, modifié, ou indexé avec une autre clé**
   (autre embedder, autre schéma) : découpe, calcule les vecteurs des
   enfants (par lots), puis **remplace** ses points dans Qdrant et OpenSearch
   (suppression par ``document_id`` puis écriture : pas de doublon) ;
3. synchronise le magasin (fiches ajoutées, remplacées, supprimées), puis y
   enregistre les morceaux et la clé d'indexation des documents traités ;
4. supprime de l'index les documents disparus et les anciennes versions des
   documents modifiés.

L'index est écrit **avant** le magasin : si le passage s'arrête au milieu,
le magasin ne note pas le document comme indexé et le passage suivant le
refait. Les documents inchangés ne sont ni redécoupés ni réembarqués.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from luciole.core.chunk import Chunk
from luciole.core.document import ParsedDocument
from luciole.core.dossier import DocumentRecord
from luciole.core.profile import UseCaseProfile
from luciole.index.embedder import Embedder, index_key
from luciole.index.opensearch import OpenSearchIndex
from luciole.index.qdrant import QdrantIndex
from luciole.ingestion.chunking import Chunker
from luciole.ingestion.dossiers import Registry, scan
from luciole.store.sqlite import MetadataStore, SyncReport


@dataclass
class PipelineReport:
    sync: SyncReport
    indexed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # déjà indexés avec la même clé
    deleted: list[str] = field(default_factory=list)  # retirés de l'index
    children: int = 0
    parents: int = 0
    seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"{len(self.indexed)} document(s) indexé(s) ({self.children} morceaux enfants, "
            f"{self.parents} parents), {len(self.skipped)} déjà à jour, "
            f"{len(self.deleted)} retiré(s) de l'index, en {self.seconds:.1f} s"
        )


class IngestionPipeline:
    def __init__(
        self,
        profile: UseCaseProfile,
        store: MetadataStore,
        registry: Registry,
        chunker: Chunker,
        embedder: Embedder,
        vectors: QdrantIndex,
        lexical: OpenSearchIndex | None = None,
        *,
        batch_size: int = 16,
    ) -> None:
        if vectors.dimension != embedder.dimension:
            raise ValueError(
                f"dimension de l'index ({vectors.dimension}) ≠ embedder ({embedder.dimension})"
            )
        self.profile = profile
        self.store = store
        self.registry = registry
        self.chunker = chunker
        self.embedder = embedder
        self.vectors = vectors
        self.lexical = lexical
        self.batch_size = max(1, batch_size)
        self.key = index_key(embedder)

    # ------------------------------------------------------------------
    def run(self, root: str | Path) -> PipelineReport:
        start = time.perf_counter()
        self.vectors.ensure()
        if self.lexical is not None:
            self.lexical.ensure()
        pending: dict[str, list[Chunk]] = {}
        skipped: list[str] = []

        def on_document(record: DocumentRecord, parsed: ParsedDocument) -> None:
            if self.store.indexed_with(record.document_id) == self.key:
                skipped.append(record.document_id)
                return
            pending[record.document_id] = self.index_document(record, parsed)

        result = scan(root, self.profile, self.registry, on_document=on_document)
        sync = self.store.sync(result)
        for document_id, chunks in pending.items():
            self.store.replace_chunks(document_id, chunks, self.key)
        deleted = sorted({*sync.removed, *sync.replaced.values()})
        for document_id in deleted:
            self.delete_document(document_id)
        report = PipelineReport(
            sync=sync,
            indexed=sorted(pending),
            skipped=sorted(skipped),
            deleted=deleted,
            children=sum(1 for cs in pending.values() for c in cs if c.kind == "child"),
            parents=sum(1 for cs in pending.values() for c in cs if c.kind == "parent"),
            seconds=time.perf_counter() - start,
        )
        logger.info(report.summary())
        return report

    def index_document(self, record: DocumentRecord, parsed: ParsedDocument) -> list[Chunk]:
        """Découpe et indexe un document ; renvoie ses morceaux (pour le magasin)."""
        chunks = self.chunker.chunk(parsed, record)
        children = [c for c in chunks if c.kind == "child"]
        vectors = {
            "content": self._embed([c.text for c in children]),
            "context": self._embed([c.embed_text for c in children]),
        }
        self.vectors.replace_document(record.document_id, children, vectors)
        if self.lexical is not None:
            self.lexical.replace_document(record.document_id, children)
        logger.debug(f"{record.relative_path} : {len(children)} enfant(s) indexé(s)")
        return chunks

    def delete_document(self, document_id: str) -> None:
        self.vectors.delete_document(document_id)
        if self.lexical is not None:
            self.lexical.delete_document(document_id)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.embedder.embed(texts[i : i + self.batch_size])
            if len(batch) != len(texts[i : i + self.batch_size]):
                raise ValueError("l'embedder a renvoyé un nombre de vecteurs inattendu")
            for vector in batch:
                if len(vector) != self.embedder.dimension:
                    raise ValueError(
                        f"vecteur de dimension {len(vector)}, attendu {self.embedder.dimension}"
                    )
            out.extend(batch)
        return out
