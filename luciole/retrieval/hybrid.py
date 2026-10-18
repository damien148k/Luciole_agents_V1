"""Outil ``retrieve`` : recherche hybride, fusion RRF, reranker, passages cités.

1. Filtre facultatif par métadonnées : le magasin structuré donne la liste des
   documents permis (``Condition`` au niveau document, ``dossier.<champ>``
   compris) ; un filtre qui ne garde aucun document rend une liste vide, il
   n'élargit jamais la recherche.
2. Une liste dense par vecteur nommé du profil (``content``, ``context``) et
   une liste BM25, chacune de ``*_top_n`` morceaux enfants.
3. Fusion RRF ; les ``rerank_candidates`` premiers passent au reranker, qui
   lit ``Chunk.embed_text`` (titre et section, puis passage).
4. Chaque passage remonte à son morceau parent (plus de contexte pour la
   réponse) ; deux enfants du même parent n'en font qu'un, le mieux classé.
5. Règles du lot (``lot.py``) : passages de sommaire écartés, documents
   secondaires plafonnés ; les places libérées vont aux candidats suivants.

Les morceaux sont relus dans le magasin : un point d'index que le magasin ne
connaît plus (index en retard) est écarté et compté, jamais rendu.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from luciole.core.chunk import Chunk
from luciole.core.profile import RetrievalSpec
from luciole.index.embedder import Embedder
from luciole.index.opensearch import OpenSearchIndex
from luciole.index.qdrant import QdrantIndex
from luciole.retrieval.fusion import rrf
from luciole.retrieval.lot import is_secondary, is_table_of_contents
from luciole.retrieval.reranker import Reranker
from luciole.store.sqlite import Condition, MetadataStore


@dataclass
class Passage:
    #: Morceau enfant trouvé par la recherche.
    hit: Chunk
    #: Ce qui est rendu : le parent si ``expand_to_parent``, sinon l'enfant.
    unit: Chunk
    rrf: float
    ranks: dict[str, int]
    rerank: float | None = None

    @property
    def text(self) -> str:
        return self.unit.text

    @property
    def citation(self) -> str:
        return self.unit.citation()

    def to_dict(self) -> dict:
        return {
            "citation": self.citation,
            "text": self.text,
            "chunk_id": self.unit.chunk_id,
            "hit_chunk_id": self.hit.chunk_id,
            "document_id": self.unit.document_id,
            "dossier_id": self.unit.dossier_id,
            "relative_path": self.unit.relative_path,
            "page_start": self.unit.page_start,
            "page_end": self.unit.page_end,
            "page_labels": list(self.unit.page_labels),
            "section_path": list(self.unit.section_path),
            "scores": {"rrf": self.rrf, "rerank": self.rerank, "ranks": dict(self.ranks)},
        }


@dataclass
class RetrievalResult:
    query: str
    passages: list[Passage]
    #: Taille de chaque liste avant fusion (« dense:content », « bm25 »…).
    lists: dict[str, int] = field(default_factory=dict)
    fused: int = 0
    reranked: int = 0
    #: Documents permis par le filtre (None = pas de filtre).
    allowed_documents: int | None = None
    #: Points d'index inconnus du magasin, écartés.
    stale: int = 0
    #: Passages de sommaire écartés, et passages secondaires au-delà du plafond.
    skipped_toc: int = 0
    capped_secondary: int = 0
    reranker: str | None = None

    def trace(self) -> dict:
        return {
            "query": self.query,
            "lists": dict(self.lists),
            "fused": self.fused,
            "reranked": self.reranked,
            "allowed_documents": self.allowed_documents,
            "stale": self.stale,
            "skipped_toc": self.skipped_toc,
            "capped_secondary": self.capped_secondary,
            "reranker": self.reranker,
            "passages": [p.to_dict() | {"text": None} for p in self.passages],
        }


class HybridRetriever:
    def __init__(
        self,
        spec: RetrievalSpec,
        embedder: Embedder,
        vectors: QdrantIndex,
        lexical: OpenSearchIndex | None,
        store: MetadataStore,
        reranker: Reranker | None,
    ) -> None:
        if spec.rerank_model is not None and reranker is None:
            raise ValueError(f"le profil demande le reranker « {spec.rerank_model} »")
        self.spec = spec
        self.embedder = embedder
        self.vectors = vectors
        self.lexical = lexical
        self.store = store
        self.reranker = reranker if spec.rerank_model is not None else None

    def _allowed(
        self, document_ids: Sequence[str] | None, conditions: list[Condition] | None
    ) -> list[str] | None:
        allowed = None if document_ids is None else list(dict.fromkeys(document_ids))
        if conditions:
            selected = [d.document_id for d in self.store.select("document", conditions)]
            keep = set(selected)
            allowed = selected if allowed is None else [d for d in allowed if d in keep]
        return allowed

    def retrieve(
        self,
        query: str,
        *,
        document_ids: Sequence[str] | None = None,
        conditions: list[Condition] | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        query = query.strip()
        if not query:
            raise ValueError("retrieve : question vide")
        spec = self.spec
        top_k = top_k or spec.rerank_top_k
        allowed = self._allowed(document_ids, conditions)
        result = RetrievalResult(
            query,
            [],
            allowed_documents=None if allowed is None else len(allowed),
            reranker=getattr(self.reranker, "name", None),
        )
        if allowed is not None and not allowed:
            return result

        lists: dict[str, list[str]] = {}
        [vector] = self.embedder.embed([query])
        for using in spec.dense_vectors:
            hits = self.vectors.search(
                vector, using=using, limit=spec.dense_top_n, document_ids=allowed
            )
            lists[f"dense:{using}"] = [h.chunk_id for h in hits]
        if self.lexical is not None:
            hits = self.lexical.search(query, limit=spec.bm25_top_n, document_ids=allowed)
            lists["bm25"] = [h.chunk_id for h in hits]
        result.lists = {name: len(ids) for name, ids in lists.items()}

        fused = rrf(lists, spec.rrf_k)
        result.fused = len(fused)
        candidates: list[tuple[Chunk, float, dict[str, int]]] = []
        for item in fused:
            if len(candidates) == spec.rerank_candidates:
                break
            chunk = self.store.get_chunk(item.chunk_id)
            if chunk is None:
                result.stale += 1
                continue
            candidates.append((chunk, item.score, item.ranks))

        scores: list[float | None] = [None] * len(candidates)
        if self.reranker is not None and candidates:
            scores = list(self.reranker.score(query, [c.embed_text for c, _, _ in candidates]))
            if len(scores) != len(candidates):
                raise ValueError(f"reranker : {len(scores)} scores pour {len(candidates)} passages")
            result.reranked = len(candidates)
            order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        else:
            order = list(range(len(candidates)))

        seen: set[str] = set()
        secondary = 0
        cap = spec.secondary_cap if spec.secondary_documents else 0
        for i in order:
            chunk, score, ranks = candidates[i]
            if spec.exclude_toc and is_table_of_contents(chunk.text):
                result.skipped_toc += 1
                continue
            if cap and is_secondary(chunk.file_name, spec.secondary_documents):
                if secondary >= cap:
                    result.capped_secondary += 1
                    continue
                secondary += 1
            unit = chunk
            if spec.expand_to_parent and chunk.parent_id:
                unit = self.store.get_chunk(chunk.parent_id) or chunk
            if unit.chunk_id in seen:
                continue
            seen.add(unit.chunk_id)
            result.passages.append(Passage(chunk, unit, score, ranks, scores[i]))
            if len(result.passages) == top_k:
                break
        return result
