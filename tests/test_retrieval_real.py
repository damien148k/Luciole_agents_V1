"""Recherche hybride avec les **vrais** modèles (machines de Damien) : BGE-M3
et le reranker bge-reranker-v2-m3 à leur révision épinglée, contre les vrais
Qdrant et OpenSearch de ``docker/compose.services.yaml``.

Si un modèle manque, le test échoue en disant de lancer
``.\\DEV.ps1 -Action models`` : c'est l'installation qu'il faut compléter, pas
le test qu'il faut sauter. GPU si PyTorch en voit un, sinon processeur.
Les questions sont des reformulations : peu de mots en commun avec les
réunions, c'est le sens qui doit trouver le passage.
"""

import shutil
import uuid
from pathlib import Path

import pytest

from luciole.core.hardware import load_hardware_profile
from luciole.core.model_files import load_manifest
from luciole.core.profile import load_profile
from luciole.index.clients import opensearch_client, qdrant_client
from luciole.index.opensearch import OpenSearchIndex
from luciole.index.pipeline import IngestionPipeline
from luciole.index.qdrant import QdrantIndex
from luciole.ingestion.chunking import Chunker
from luciole.ingestion.registry import default_registry
from luciole.ingestion.tokenizer import load_tokenizer
from luciole.retrieval.hybrid import HybridRetriever
from luciole.store.sqlite import Condition, MetadataStore

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
VENTES = load_profile(ROOT / "configs" / "profiles" / "ventes").profile
MANIFEST = load_manifest(ROOT / "configs" / "models.yaml")


@pytest.fixture(scope="module")
def hardware():
    import torch

    machine = "rtx3080ti" if torch.cuda.is_available() else "cpu"
    print(f"\nModèles sur le profil {machine}")
    return load_hardware_profile(ROOT / "configs" / "hardware" / f"{machine}.yaml")


@pytest.fixture(scope="module")
def reranker(hardware):
    from luciole.retrieval.reranker import LocalReranker

    return LocalReranker(MANIFEST.get(VENTES.retrieval.rerank_model), hardware)


@pytest.fixture(scope="module")
def indexed(hardware, reranker, tmp_path_factory):
    from luciole.index.local_embedder import LocalEmbedder

    embedder = LocalEmbedder(MANIFEST.get(VENTES.embedding.model), hardware)
    qdrant, opensearch = qdrant_client(), opensearch_client()
    try:
        qdrant.get_collections()
        opensearch.info()
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"services injoignables ({e}) : « .\\DEV.ps1 -Action test-integration »")
    tmp = tmp_path_factory.mktemp("recherche")
    root = tmp / "ventes"
    shutil.copytree(ROOT / "tests" / "corpus" / "ventes", root)
    name = f"luciole_test_{uuid.uuid4().hex[:8]}"
    vectors = QdrantIndex(qdrant, name, embedder.dimension)
    lexical = OpenSearchIndex(opensearch, name)
    store = MetadataStore(tmp / "ventes.db", VENTES)
    IngestionPipeline(
        VENTES,
        store,
        default_registry(),
        Chunker(VENTES.chunking, load_tokenizer(VENTES.chunking.tokenizer)),
        embedder,
        vectors,
        lexical,
        batch_size=embedder.batch_size,
    ).run(root)
    yield HybridRetriever(VENTES.retrieval, embedder, vectors, lexical, store, reranker)
    store.close()
    qdrant.delete_collection(name)
    opensearch.indices.delete(index=name, ignore_unavailable=True)


def test_reranker_note_le_sens(reranker):
    assert reranker.name.startswith("BAAI/bge-reranker-v2-m3@")
    good, bad = reranker.score(
        "Pourquoi le client a-t-il renoncé à acheter ?",
        [
            "Douze mille euros la première année, c'est au-dessus de ce qu'on peut engager.",
            "L'installation se fait en une demi-journée sur votre serveur.",
        ],
    )
    assert good > bad


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "Pourquoi l'atelier a-t-il repoussé son achat à l'an prochain ?",
            "atelier-dubois/reunion-3.transcript.md",
        ),
        (
            "À quelle condition le cabinet accepte-t-il de signer ?",
            "cabinet-roche/reunion-2.transcript.md",
        ),
    ],
)
def test_bon_passage_en_tete(indexed, question, expected):
    result = indexed.retrieve(question)
    print(f"\n{question}")
    for p in result.passages[:3]:
        print(f"  {p.citation} (RRF {p.rrf:.4f}, reranker {p.rerank:.3f})")
    assert result.reranked > 0 and result.passages[0].unit.relative_path == expected


def test_filtre_puis_recherche(indexed):
    result = indexed.retrieve(
        "Pourquoi l'atelier a-t-il repoussé son achat à l'an prochain ?",
        conditions=[Condition("dossier.issue", "eq", "gagne")],
    )
    assert result.passages and {p.unit.dossier_id for p in result.passages} == {"cabinet-roche"}
