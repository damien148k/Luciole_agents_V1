"""Index vectoriel Qdrant : un point par morceau **enfant**, deux vecteurs
nommés.

- ``content`` : le passage seul ;
- ``context`` : « Titre > Section » puis le passage (``Chunk.embed_text``).

V5 n'avait qu'un vecteur sans nom qui mélangeait les deux, et réutilisait une
collection existante sans en vérifier la dimension (``analyse-v5.md``, 5.1).
Ici la collection est créée avec ses deux vecteurs nommés, et une collection
existante de forme différente est refusée. Le client est injecté :
``QdrantClient(":memory:")`` dans les tests, le serveur en production.
"""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from luciole.core.chunk import Chunk

VECTORS = ("content", "context")
_KEYWORD_FIELDS = ("document_id", "dossier_id", "chunk_id", "parent_id", "doc_type")
# Points par requête d'écriture. Qdrant refuse un corps JSON de plus de 32 Mo :
# le 2026-10-18, un tome de 396 pages envoyé d'un bloc pesait 56 Mo. Un point
# (deux vecteurs de 1024 nombres et son texte) pèse environ 50 ko en JSON.
SEND_SIZE = 128


def point_id(chunk_id: str) -> str:
    """Qdrant n'accepte que des entiers ou des UUID : UUID déterministe."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"luciole:{chunk_id}"))


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float
    payload: dict


class QdrantIndex:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        dimension: int,
        *,
        send_size: int = SEND_SIZE,
    ) -> None:
        if send_size < 1:
            raise ValueError(f"send_size doit valoir au moins 1, pas {send_size}")
        self.client = client
        self.collection = collection
        self.dimension = dimension
        self.send_size = send_size

    def ensure(self) -> None:
        """Crée la collection si besoin ; refuse une collection de forme différente."""
        if self.client.collection_exists(self.collection):
            vectors = self.client.get_collection(self.collection).config.params.vectors
            shape = (
                {name: (p.size, p.distance) for name, p in vectors.items()}
                if isinstance(vectors, dict)
                else {}
            )
            expected = {name: (self.dimension, models.Distance.COSINE) for name in VECTORS}
            if shape != expected:
                raise ValueError(
                    f"collection Qdrant « {self.collection} » incompatible : {shape} "
                    f"au lieu de {expected}"
                )
            return
        self.client.create_collection(
            self.collection,
            vectors_config={
                name: models.VectorParams(size=self.dimension, distance=models.Distance.COSINE)
                for name in VECTORS
            },
        )
        with warnings.catch_warnings():  # sans effet (et signalé) en mode local
            warnings.simplefilter("ignore")
            for name in _KEYWORD_FIELDS:
                self.client.create_payload_index(
                    self.collection, name, models.PayloadSchemaType.KEYWORD
                )

    def _document_filter(self, document_id: str) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))
            ]
        )

    def delete_document(self, document_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=self._document_filter(document_id)),
            wait=True,
        )

    def replace_document(
        self,
        document_id: str,
        chunks: Sequence[Chunk],
        vectors: dict[str, Sequence[Sequence[float]]],
    ) -> None:
        """Supprime les points du document puis écrit les nouveaux : réindexer
        deux fois le même document ne crée pas de doublon. Les points partent
        par paquets de ``send_size`` pour rester sous la limite de Qdrant."""
        for name in VECTORS:
            if len(vectors[name]) != len(chunks):
                raise ValueError(f"vecteurs « {name} » : {len(vectors[name])} pour {len(chunks)}")
        self.delete_document(document_id)
        if not chunks:
            return
        points = [
            models.PointStruct(
                id=point_id(c.chunk_id),
                vector={name: list(vectors[name][i]) for name in VECTORS},
                payload=payload(c),
            )
            for i, c in enumerate(chunks)
        ]
        for start in range(0, len(points), self.send_size):
            batch = points[start : start + self.send_size]
            self.client.upsert(self.collection, points=batch, wait=True)

    def count(self, document_id: str | None = None) -> int:
        flt = self._document_filter(document_id) if document_id else None
        return self.client.count(self.collection, count_filter=flt, exact=True).count

    def search(
        self,
        vector: Sequence[float],
        *,
        using: str = "context",
        limit: int = 10,
        document_ids: Sequence[str] | None = None,
    ) -> list[Hit]:
        flt = None
        if document_ids is not None:
            flt = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchAny(any=list(document_ids))
                    )
                ]
            )
        result = self.client.query_points(
            self.collection, query=list(vector), using=using, limit=limit, query_filter=flt
        )
        return [Hit(p.payload["chunk_id"], p.score, p.payload) for p in result.points]


def payload(chunk: Chunk) -> dict:
    """Ce que l'index garde d'un morceau : de quoi citer et filtrer sans relire
    le magasin (texte, document, dossier, pages, numéros imprimés, versions)."""
    data = chunk.model_dump(mode="json")
    data["section_path"] = list(chunk.section_path)
    data["citation"] = chunk.citation()
    return data
