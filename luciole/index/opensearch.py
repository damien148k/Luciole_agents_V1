"""Index lexical OpenSearch (BM25) : un document par morceau **enfant**.

Le texte et le contexte sont analysés en français (racines, mots vides) ; les
identifiants sont des mots-clés exacts pour supprimer et filtrer. Les
métadonnées du document sont gardées sans être indexées : on filtre par
document via le magasin structuré (voir ``DECISIONS.md``). Le client est
injecté : un faux en mémoire dans les tests, le serveur en production.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from luciole.core.chunk import Chunk
from luciole.index.qdrant import Hit, payload

MAPPING: dict[str, Any] = {
    "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "chunk_id": {"type": "keyword"},
            "kind": {"type": "keyword"},
            "parent_id": {"type": "keyword"},
            "index": {"type": "integer"},
            "document_id": {"type": "keyword"},
            "dossier_id": {"type": "keyword"},
            "relative_path": {"type": "keyword"},
            "file_name": {"type": "keyword"},
            "doc_type": {"type": "keyword"},
            "content_hash": {"type": "keyword"},
            "text": {"type": "text", "analyzer": "french"},
            "context": {"type": "text", "analyzer": "french"},
            "token_count": {"type": "integer"},
            "page_start": {"type": "integer"},
            "page_end": {"type": "integer"},
            "page_labels": {"type": "keyword"},
            "section_path": {"type": "keyword"},
            "node_ids": {"type": "keyword", "index": False},
            "methods": {"type": "keyword"},
            "metadata": {"type": "object", "enabled": False},
            "versions": {"type": "object", "enabled": False},
            "citation": {"type": "keyword", "index": False},
        },
    },
}


class OpenSearchIndex:
    def __init__(self, client: Any, index: str) -> None:
        self.client = client
        self.index = index

    def ensure(self) -> None:
        if not self.client.indices.exists(index=self.index):
            self.client.indices.create(index=self.index, body=MAPPING)

    def delete_document(self, document_id: str) -> None:
        self.client.delete_by_query(
            index=self.index,
            body={"query": {"term": {"document_id": document_id}}},
            refresh=True,
            conflicts="proceed",
        )

    def replace_document(self, document_id: str, chunks: Sequence[Chunk]) -> None:
        self.delete_document(document_id)
        if not chunks:
            return
        body: list[dict] = []
        for c in chunks:
            body.append({"index": {"_index": self.index, "_id": c.chunk_id}})
            body.append(payload(c))
        response = self.client.bulk(body=body, refresh=True)
        if response.get("errors"):
            failed = [
                item["index"].get("error")
                for item in response.get("items", [])
                if item.get("index", {}).get("error")
            ]
            raise RuntimeError(f"OpenSearch : {len(failed)} morceau(x) refusé(s) : {failed[:2]}")

    def count(self, document_id: str | None = None) -> int:
        body = {"query": {"term": {"document_id": document_id}}} if document_id else None
        return int(self.client.count(index=self.index, body=body)["count"])

    def search(
        self, query: str, *, limit: int = 10, document_ids: Sequence[str] | None = None
    ) -> list[Hit]:
        must: list[dict] = [{"multi_match": {"query": query, "fields": ["text^2", "context"]}}]
        flt = [{"terms": {"document_id": list(document_ids)}}] if document_ids is not None else []
        response = self.client.search(
            index=self.index,
            body={"size": limit, "query": {"bool": {"must": must, "filter": flt}}},
        )
        return [
            Hit(h["_source"]["chunk_id"], float(h["_score"]), h["_source"])
            for h in response["hits"]["hits"]
        ]
