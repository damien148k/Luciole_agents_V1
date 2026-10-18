"""Porte de l'étape 3 contre de **vrais** serveurs Qdrant et OpenSearch
(``docker/compose.services.yaml``, lancés par ``DEV.ps1 -Action
test-integration``). L'embedder est encore le faux : le vrai BGE-M3 arrive
avec la PR 3b. Les index de test portent un nom à part et sont supprimés.
"""

import shutil
import sys
import uuid
from pathlib import Path

import pytest

from luciole.core.profile import ChunkingSpec, load_profile
from luciole.index.clients import opensearch_client, qdrant_client
from luciole.index.opensearch import OpenSearchIndex
from luciole.index.pipeline import IngestionPipeline
from luciole.index.qdrant import QdrantIndex
from luciole.ingestion.chunking import Chunker
from luciole.ingestion.registry import default_registry
from luciole.store.sqlite import MetadataStore

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from fakes import FakeEmbedder  # noqa: E402
from test_chunking import FakeCounter  # noqa: E402

VENTES = load_profile(ROOT / "configs" / "profiles" / "ventes").profile


@pytest.fixture()
def services(tmp_path):
    qdrant, opensearch = qdrant_client(), opensearch_client()
    try:
        qdrant.get_collections()
        opensearch.info()
    except Exception as e:  # noqa: BLE001
        pytest.fail(
            f"services d'index injoignables ({e}). Lancer Docker Desktop puis "
            "« .\\DEV.ps1 -Action test-integration », qui démarre Qdrant et OpenSearch."
        )
    name = f"luciole_test_{uuid.uuid4().hex[:8]}"
    embedder = FakeEmbedder()
    vectors = QdrantIndex(qdrant, name, embedder.dimension)
    lexical = OpenSearchIndex(opensearch, name)
    root = tmp_path / "ventes"
    shutil.copytree(ROOT / "tests" / "corpus" / "ventes", root)
    store = MetadataStore(tmp_path / "ventes.db", VENTES)
    pipeline = IngestionPipeline(
        VENTES,
        store,
        default_registry(),
        Chunker(
            ChunkingSpec(size_tokens=64, parent_size_tokens=160, max_tokens=200), FakeCounter()
        ),
        embedder,
        vectors,
        lexical,
    )
    yield root, store, vectors, lexical, pipeline, embedder
    store.close()
    qdrant.delete_collection(name)
    opensearch.indices.delete(index=name, ignore_unavailable=True)


def test_ingestion_reingestion_suppression_sur_vrais_serveurs(services):
    root, store, vectors, lexical, pipeline, embedder = services
    first = pipeline.run(root)
    children = store.count_chunks("child")
    assert len(first.indexed) == 5 and vectors.count() == lexical.count() == children > 0

    again = pipeline.run(root)
    assert again.indexed == [] and vectors.count() == lexical.count() == children

    shutil.rmtree(root / "cabinet-roche")
    gone = pipeline.run(root)
    assert len(gone.deleted) == 2
    left = store.count_chunks("child")
    assert vectors.count() == lexical.count() == left < children
    for document_id in gone.deleted:
        assert vectors.count(document_id) == lexical.count(document_id) == 0

    query = "douze mille euros au-dessus de ce qu'on peut engager"
    [vector] = embedder.embed([query])
    assert (
        vectors.search(vector, using="content", limit=1)[0]
        .payload["relative_path"]
        .endswith("reunion-3.transcript.md")
    )
    # Vrai BM25 avec l'analyseur français : « engager » retrouve « engager ».
    assert (
        lexical.search(query, limit=1)[0]
        .payload["relative_path"]
        .endswith("reunion-3.transcript.md")
    )
