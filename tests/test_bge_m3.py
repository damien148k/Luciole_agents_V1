"""Porte de l'étape 3 avec le **vrai** BGE-M3 (machines de Damien).

Le modèle est lu dans ``LUCIOLE_MODELS_DIR`` à la révision épinglée de
``configs/models.yaml`` ; s'il manque, le test échoue en disant de lancer
``.\\DEV.ps1 -Action models`` : c'est l'installation qu'il faut compléter,
pas le test qu'il faut sauter. GPU si PyTorch en voit un, sinon processeur
(PyTorch de PyPI sous Windows). Qdrant et OpenSearch sont les vrais serveurs
de ``docker/compose.services.yaml``.
"""

import math
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
from luciole.store.sqlite import MetadataStore

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
VENTES = load_profile(ROOT / "configs" / "profiles" / "ventes").profile


@pytest.fixture(scope="module")
def bge_m3():
    import torch

    from luciole.index.local_embedder import LocalEmbedder

    machine = "rtx3080ti" if torch.cuda.is_available() else "cpu"
    hardware = load_hardware_profile(ROOT / "configs" / "hardware" / f"{machine}.yaml")
    spec = load_manifest(ROOT / "configs" / "models.yaml").get(VENTES.embedding.model)
    print(f"\nBGE-M3 sur {hardware.device} (profil {machine})")
    return LocalEmbedder(spec, hardware)


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_vecteurs_bge_m3(bge_m3):
    assert bge_m3.dimension == 1024 and bge_m3.name.startswith("BAAI/bge-m3@")
    question, eolien, vente = bge_m3.embed(
        [
            "Quel est l'impact du parc éolien sur les oiseaux migrateurs ?",
            "Les éoliennes peuvent provoquer des collisions avec l'avifaune en migration.",
            "Le client trouve le prix trop élevé pour son budget de l'année.",
        ]
    )
    assert all(math.isclose(_cos(v, v), 1.0, abs_tol=1e-4) for v in (question, eolien, vente))
    assert _cos(question, eolien) > _cos(question, vente) + 0.1
    # Même texte, même vecteur : la réingestion ne change rien.
    [again] = bge_m3.embed(["Quel est l'impact du parc éolien sur les oiseaux migrateurs ?"])
    assert _cos(question, again) > 0.999


def test_porte_etape_3_avec_bge_m3(bge_m3, tmp_path):
    qdrant, opensearch = qdrant_client(), opensearch_client()
    try:
        qdrant.get_collections()
        opensearch.info()
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"services injoignables ({e}) : « .\\DEV.ps1 -Action test-integration »")
    name = f"luciole_test_{uuid.uuid4().hex[:8]}"
    root = tmp_path / "ventes"
    shutil.copytree(ROOT / "tests" / "corpus" / "ventes", root)
    vectors = QdrantIndex(qdrant, name, bge_m3.dimension)
    lexical = OpenSearchIndex(opensearch, name)
    store = MetadataStore(tmp_path / "ventes.db", VENTES)
    pipeline = IngestionPipeline(
        VENTES,
        store,
        default_registry(),
        Chunker(VENTES.chunking, load_tokenizer(VENTES.chunking.tokenizer)),
        bge_m3,
        vectors,
        lexical,
        batch_size=bge_m3.batch_size,
    )
    try:
        first = pipeline.run(root)
        print(first.summary())
        children = store.count_chunks("child")
        assert len(first.indexed) == 5 and vectors.count() == lexical.count() == children > 0

        again = pipeline.run(root)
        assert again.indexed == [] and vectors.count() == lexical.count() == children

        [query] = bge_m3.embed(["douze mille euros au-dessus de ce qu'on peut engager"])
        top = vectors.search(query, limit=1)[0].payload
        assert top["relative_path"] == "atelier-dubois/reunion-3.transcript.md"

        shutil.rmtree(root / "cabinet-roche")
        gone = pipeline.run(root)
        assert len(gone.deleted) == 2
        assert vectors.count() == lexical.count() == store.count_chunks("child") < children
    finally:
        store.close()
        qdrant.delete_collection(name)
        opensearch.indices.delete(index=name, ignore_unavailable=True)
