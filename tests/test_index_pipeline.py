"""Pipeline d'ingestion : magasin, Qdrant (mode mémoire) et OpenSearch (faux)
restent alignés quand on ajoute, réingère, modifie et supprime des fichiers.

Porte de l'étape 3 côté code : ingestion d'un dossier de test, réingestion
sans doublon, suppression propre. La même chose contre de vrais serveurs est
dans ``test_index_services.py`` (marqueur ``integration``).
"""

import shutil
import sys
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from luciole.core.profile import ChunkingSpec, load_profile
from luciole.index.embedder import index_key, index_name
from luciole.index.opensearch import MAPPING, OpenSearchIndex
from luciole.index.pipeline import IngestionPipeline
from luciole.index.qdrant import QdrantIndex, point_id
from luciole.ingestion.chunking import Chunker
from luciole.ingestion.registry import default_registry
from luciole.store.sqlite import MetadataStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from fakes import FakeEmbedder, FakeOpenSearch  # noqa: E402
from test_chunking import FakeCounter  # noqa: E402

VENTES = load_profile(ROOT / "configs" / "profiles" / "ventes").profile
SPEC = ChunkingSpec(size_tokens=64, overlap_pct=20, parent_size_tokens=160, max_tokens=200)


class Setup:
    """Une racine copiée du corpus, un magasin et deux index neufs."""

    def __init__(self, tmp_path: Path, embedder: FakeEmbedder | None = None) -> None:
        self.root = tmp_path / "ventes"
        shutil.copytree(ROOT / "tests" / "corpus" / "ventes", self.root)
        self.store = MetadataStore(tmp_path / "ventes.db", VENTES)
        self.qdrant = QdrantClient(":memory:")
        self.opensearch = FakeOpenSearch()
        self.embedder = embedder or FakeEmbedder()
        self.pipeline = self.make(self.embedder)

    def make(self, embedder) -> IngestionPipeline:
        name = index_name("ventes", embedder)
        self.vectors = QdrantIndex(self.qdrant, name, embedder.dimension)
        self.lexical = OpenSearchIndex(self.opensearch, name)
        return IngestionPipeline(
            VENTES,
            self.store,
            default_registry(),
            Chunker(SPEC, FakeCounter()),
            embedder,
            self.vectors,
            self.lexical,
            batch_size=4,
        )

    def assert_aligned(self) -> None:
        """Chaque enfant du magasin a son point Qdrant et son document
        OpenSearch, et rien d'autre n'est dans l'index."""
        children = self.store.count_chunks("child")
        assert self.vectors.count() == self.lexical.count() == children > 0
        for doc in self.store.select("document"):
            kids = self.store.chunks_of(doc.document_id, "child")
            assert self.vectors.count(doc.document_id) == len(kids)
            assert self.lexical.count(doc.document_id) == len(kids)
            ids = [point_id(c.chunk_id) for c in kids]
            assert len(self.qdrant.retrieve(self.vectors.collection, ids)) == len(kids)


@pytest.fixture()
def env(tmp_path) -> Setup:
    s = Setup(tmp_path)
    yield s
    s.store.close()


def test_ingestion_d_un_dossier_de_test(env):
    report = env.pipeline.run(env.root)
    assert len(report.indexed) == 5 and report.skipped == [] and report.deleted == []
    assert report.children == env.store.count_chunks("child")
    assert report.parents == env.store.count_chunks("parent")
    env.assert_aligned()
    doc = env.store.select("document")[0]
    assert env.store.indexed_with(doc.document_id) == index_key(env.embedder)


def test_reingestion_sans_doublon_ni_calcul(env):
    env.pipeline.run(env.root)
    before = (env.vectors.count(), env.embedder.texts)
    report = env.pipeline.run(env.root)
    assert report.indexed == [] and len(report.skipped) == 5
    assert (env.vectors.count(), env.embedder.texts) == before, "rien recalculé"
    env.assert_aligned()


def test_fichier_modifie_reindexe_seul(env):
    env.pipeline.run(env.root)
    path = env.root / "cabinet-roche" / "reunion-2.transcript.md"
    old = next(
        d
        for d in env.store.select("document")
        if d.relative_path.endswith("roche/reunion-2.transcript.md")
    )
    path.write_text(
        path.read_text(encoding="utf-8") + "\n[00:05:00] Marc Faure : Signature lundi.\n",
        encoding="utf-8",
    )
    report = env.pipeline.run(env.root)
    assert len(report.indexed) == 1 and len(report.skipped) == 4
    assert report.deleted == [old.document_id]
    assert env.vectors.count(old.document_id) == env.lexical.count(old.document_id) == 0
    assert env.store.chunks_of(old.document_id) == []
    env.assert_aligned()
    hits = env.lexical.search("signature lundi")
    assert hits and hits[0].payload["relative_path"] == "cabinet-roche/reunion-2.transcript.md"


def test_suppression_propre(env):
    env.pipeline.run(env.root)
    roche = [d.document_id for d in env.store.select("document") if d.dossier_id == "cabinet-roche"]
    shutil.rmtree(env.root / "cabinet-roche")
    (env.root / "atelier-dubois" / "reunion-1.transcript.md").unlink()
    report = env.pipeline.run(env.root)
    assert len(report.deleted) == 3 and set(roche) <= set(report.deleted)
    for document_id in report.deleted:
        assert env.vectors.count(document_id) == env.lexical.count(document_id) == 0
    assert len(env.store.select("document")) == 2
    env.assert_aligned()


def test_changement_d_embedder_reindexe_tout(env):
    env.pipeline.run(env.root)
    other = FakeEmbedder(name="autre-modele")
    report = env.make(other).run(env.root)
    assert len(report.indexed) == 5, "clé d'index différente : tout est refait"
    assert env.vectors.collection != index_name("ventes", env.embedder)
    env.assert_aligned()


def test_arret_en_cours_puis_reprise(tmp_path):
    class Flaky(FakeEmbedder):
        def __init__(self) -> None:
            super().__init__()
            self.fail_after = 3

        def embed(self, texts):
            if self.fail_after is not None and self.calls >= self.fail_after:
                raise RuntimeError("GPU indisponible")
            return super().embed(texts)

    env = Setup(tmp_path, Flaky())
    with pytest.raises(RuntimeError, match="GPU"):
        env.pipeline.run(env.root)
    assert env.store.count_chunks() == 0, "rien noté comme indexé"
    env.embedder.fail_after = None
    report = env.pipeline.run(env.root)
    assert len(report.indexed) == 5
    env.assert_aligned()  # les points écrits avant l'arrêt n'ont pas fait de doublon
    env.store.close()


def test_recherche_dense_et_lexicale(env):
    env.pipeline.run(env.root)
    query = "douze mille euros au-dessus de ce qu'on peut engager"
    [vector] = env.embedder.embed([query])
    dense = env.vectors.search(vector, using="content", limit=3)
    lexical = env.lexical.search(query, limit=3)
    for hits in (dense, lexical):
        top = hits[0].payload
        assert top["relative_path"] == "atelier-dubois/reunion-3.transcript.md"
        assert top["citation"].startswith("reunion-3.transcript.md")
        assert top["metadata"]["numero_reunion"] == 3
    roche = [d.document_id for d in env.store.select("document") if d.dossier_id == "cabinet-roche"]
    filtered = env.vectors.search(vector, limit=10, document_ids=roche)
    assert filtered and {h.payload["dossier_id"] for h in filtered} == {"cabinet-roche"}


def test_parent_relu_dans_le_magasin(env):
    env.pipeline.run(env.root)
    [vector] = env.embedder.embed(["démonstration sur l'ancien dossier du gymnase"])
    hit = env.vectors.search(vector, limit=1)[0]
    parent = env.store.get_chunk(hit.payload["parent_id"])
    assert parent.kind == "parent" and hit.payload["text"].split("\n\n")[0] in parent.text
    chunks = env.store.chunks_of(parent.document_id)
    assert chunks[0].kind == "parent" and chunks[-1].kind == "child"


def test_dimension_et_collection_incompatibles(env):
    with pytest.raises(ValueError, match="dimension"):
        IngestionPipeline(
            VENTES,
            env.store,
            default_registry(),
            Chunker(SPEC, FakeCounter()),
            FakeEmbedder(dimension=32),
            env.vectors,
        )
    env.vectors.ensure()
    with pytest.raises(ValueError, match="incompatible"):
        QdrantIndex(env.qdrant, env.vectors.collection, 128).ensure()


def test_nom_d_index_stable_et_propre_au_modele():
    a, b = FakeEmbedder(), FakeEmbedder(name="autre")
    assert index_name("mrae", a) == index_name("mrae", FakeEmbedder())
    assert index_name("mrae", a) != index_name("mrae", b) != index_name("ventes", b)
    assert index_name("mrae", a).startswith("luciole_mrae_v1_")


def test_correspondance_opensearch_stricte():
    """Le payload d'un morceau entre exactement dans la correspondance déclarée
    (``dynamic: strict`` refuserait un champ oublié)."""
    from luciole.core.chunk import Chunk
    from luciole.index.qdrant import payload

    chunk = Chunk(
        chunk_id="d:c00000",
        kind="child",
        parent_id="d:p00000",
        index=0,
        document_id="d",
        dossier_id="x",
        relative_path="x/a.md",
        file_name="a.md",
        doc_type="md",
        content_hash="h",
        text="Bonjour.",
        token_count=4,
    )
    assert set(payload(chunk)) == set(MAPPING["mappings"]["properties"])


def test_magasin_refuse_des_morceaux_orphelins(env):
    with pytest.raises(KeyError):
        env.store.replace_chunks("inconnu", [], "cle")
