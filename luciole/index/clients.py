"""Connexion aux services d'index, adresses lues dans l'environnement.

Par défaut, les services lancés par ``docker/compose.services.yaml`` sur la
machine locale. ``LUCIOLE_QDRANT_URL=:memory:`` donne un Qdrant en mémoire
(sans serveur), utile pour essayer le pipeline.
"""

from __future__ import annotations

import os

DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_OPENSEARCH_URL = "http://127.0.0.1:9200"


def qdrant_client(url: str | None = None):
    from qdrant_client import QdrantClient

    url = url or os.environ.get("LUCIOLE_QDRANT_URL", DEFAULT_QDRANT_URL)
    if url == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(url=url, timeout=30)


def opensearch_client(url: str | None = None):
    from opensearchpy import OpenSearch

    url = url or os.environ.get("LUCIOLE_OPENSEARCH_URL", DEFAULT_OPENSEARCH_URL)
    return OpenSearch(hosts=[url], timeout=30)
