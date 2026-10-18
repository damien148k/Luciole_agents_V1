"""Watcher : comportements des tests de V5 (``tests/watcher/``), portés sur
l'architecture à un seul fil.

Correspondance :
- observer (temporaires, extensions, debounce) → ``snapshot`` et stabilité ;
- reconciler (nouveau, modifié, supprimé, inchangé) → passages du pipeline ;
- crash_recovery → passage en échec repris, index jamais dédoublé ;
- queue_concurrency (panne du 3 août 2026) → rafale de suppressions traitée
  en un passage, un seul watcher par magasin.
"""

import shutil
import sys
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from luciole.index.opensearch import OpenSearchIndex
from luciole.index.qdrant import QdrantIndex
from luciole.index.watcher import Watcher, WatcherLock, WatchSettings, main, snapshot
from luciole.ingestion.registry import default_registry

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from fakes import FakeEmbedder, FakeOpenSearch  # noqa: E402
from test_index_pipeline import VENTES, Setup  # noqa: E402

REGISTRY = default_registry()


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class Runs:
    """Faux pipeline : compte les passages, échoue sur demande."""

    def __init__(self) -> None:
        self.count = 0
        self.fail = 0

    def __call__(self) -> None:
        self.count += 1
        if self.fail:
            self.fail -= 1
            raise RuntimeError("Qdrant injoignable")


@pytest.fixture()
def root(tmp_path) -> Path:
    root = tmp_path / "ventes"
    shutil.copytree(ROOT / "tests" / "corpus" / "ventes", root)
    return root


def watcher_on(root: Path, runs: Runs, clock: Clock, interval: float = 5.0) -> Watcher:
    return Watcher(
        runs,
        lambda: snapshot(root, VENTES, REGISTRY),
        WatchSettings(interval=interval, max_backoff=60.0),
        clock=clock,
        sleep=clock.sleep,
    )


# --- relevé (observer de V5) -------------------------------------------------
def test_releve_ignore_temporaires_formats_et_racine(root):
    before = snapshot(root, VENTES, REGISTRY)
    assert "cabinet-roche/dossier.yaml" in before and len(before) == 6
    (root / "atelier-dubois" / "~$reunion-4.docx").write_bytes(b"verrou Word")
    (root / "atelier-dubois" / ".~lock.reunion-4.odt#").write_text("verrou LibreOffice")
    (root / "atelier-dubois" / "reunion-4.md.crdownload").write_text("téléchargement")
    (root / "atelier-dubois" / ".cache").mkdir()
    (root / "atelier-dubois" / ".cache" / "note.md").write_text("caché")
    (root / "a-la-racine.md").write_text("hors dossier")
    (root / "atelier-dubois" / "Thumbs.db").write_bytes(b"\0")
    assert snapshot(root, VENTES, REGISTRY) == before


def test_releve_voit_ajout_modification_suppression(root):
    before = snapshot(root, VENTES, REGISTRY)
    note = root / "atelier-dubois" / "notes.md"
    note.write_text("Rappeler le client.", encoding="utf-8")
    added = snapshot(root, VENTES, REGISTRY)
    assert set(added) - set(before) == {"atelier-dubois/notes.md"}
    note.write_text("Rappeler le client mardi.", encoding="utf-8")
    assert snapshot(root, VENTES, REGISTRY) != added
    note.unlink()
    assert snapshot(root, VENTES, REGISTRY) == before
    assert snapshot(root / "absente", VENTES, REGISTRY) == {}


# --- déclenchement (debounce de V5) -------------------------------------------
def test_premier_releve_reconcilie_puis_rien_sans_changement(root):
    runs, clock = Runs(), Clock()
    w = watcher_on(root, runs, clock)
    assert w.tick() and runs.count == 1, "un passage au démarrage (réconciliation)"
    for _ in range(3):
        assert not w.tick()
    assert runs.count == 1


def test_attend_un_releve_stable_et_fusionne_les_changements_rapides(root):
    runs, clock = Runs(), Clock()
    w = watcher_on(root, runs, clock)
    w.tick()
    copy = root / "atelier-dubois" / "reunion-4.transcript.md"
    copy.write_text("[00:00:10] Camille Martin : Bonjour", encoding="utf-8")
    assert not w.tick(), "fichier vu une fois : peut-être encore en copie"
    copy.write_text("[00:00:10] Camille Martin : Bonjour, on signe ?", encoding="utf-8")
    (root / "atelier-dubois" / "notes.md").write_text("Relancer.", encoding="utf-8")
    assert not w.tick(), "encore des changements"
    assert w.tick() and runs.count == 2, "relevé stable : un seul passage pour tout"
    assert not w.tick()


def test_rafale_de_suppressions_un_seul_passage(root):
    """La panne de V5 : plusieurs suppressions simultanées. Ici, une rafale
    donne un relevé, donc un passage, sans file ni fil concurrent."""
    runs, clock = Runs(), Clock()
    w = watcher_on(root, runs, clock)
    w.tick()
    shutil.rmtree(root / "cabinet-roche")
    for name in ("reunion-1", "reunion-2", "reunion-3"):
        (root / "atelier-dubois" / f"{name}.transcript.md").unlink()
    w.tick()
    w.tick()
    assert runs.count == 2 and not w.tick()


# --- reprise après échec (crash_recovery de V5) --------------------------------
def test_echec_retente_avec_attente_croissante(root):
    runs, clock = Runs(), Clock()
    runs.fail = 2
    w = watcher_on(root, runs, clock, interval=5.0)
    assert w.tick() and w.failures == 1 and w.retry_at == 10.0
    clock.now = 9.0
    assert not w.tick(), "attente de 2 × l'intervalle"
    clock.now = 10.0
    assert w.tick() and w.failures == 2 and w.retry_at == 30.0
    clock.now = 30.0
    assert w.tick() and w.failures == 0 and runs.count == 3
    assert not w.tick()


def test_attente_plafonnee(root):
    runs, clock = Runs(), Clock()
    runs.fail = 10
    w = watcher_on(root, runs, clock, interval=5.0)
    for _ in range(10):
        clock.now = w.retry_at
        w.tick()
    assert w.retry_at - clock.now == 60.0


def test_boucle_survit_aux_echecs_et_s_arrete(root):
    runs, clock = Runs(), Clock()
    runs.fail = 1
    w = watcher_on(root, runs, clock, interval=5.0)
    w.loop(max_ticks=4)
    assert runs.count == 2 and w.failures == 0 and clock.now == 15.0
    w.stop()
    w.loop()
    assert runs.count == 2


# --- avec le vrai pipeline (reconciler de V5) ----------------------------------
def test_watcher_et_pipeline_restent_alignes(tmp_path):
    env = Setup(tmp_path)
    clock = Clock()
    w = Watcher(
        lambda: env.pipeline.run(env.root),
        lambda: snapshot(env.root, VENTES, REGISTRY),
        WatchSettings(interval=5.0),
        clock=clock,
        sleep=clock.sleep,
    )
    w.tick()
    env.assert_aligned()
    computed = env.embedder.texts

    path = env.root / "atelier-dubois" / "reunion-3.transcript.md"
    text = path.read_text(encoding="utf-8")
    assert "issue: perdu" in text
    path.write_text(text.replace("issue: perdu", "issue: gagne"), encoding="utf-8")
    (env.root / "cabinet-roche" / "reunion-1.transcript.md").unlink()
    w.tick()
    w.tick()
    env.assert_aligned()
    assert env.store.get_dossier("atelier-dubois").metadata["issue"] == "gagne"
    assert len(env.store.select("document")) == 4
    assert env.embedder.texts > computed
    env.store.close()


# --- un seul watcher par magasin (queue_concurrency de V5) ----------------------
def test_un_seul_watcher_par_magasin(tmp_path):
    store = tmp_path / "data" / "ventes.db"
    with WatcherLock(store):
        with pytest.raises(RuntimeError, match="un autre watcher"):
            WatcherLock(store).acquire()
    with WatcherLock(store):  # libéré à la sortie
        pass


def test_ligne_de_commande(tmp_path, root, monkeypatch, capsys):
    from test_chunking import _save_tokenizer

    monkeypatch.setenv("LUCIOLE_MODELS_DIR", str(tmp_path / "modeles"))
    _save_tokenizer(tmp_path / "modeles" / "BAAI" / "bge-m3", max_length=8192)
    qdrant, opensearch = QdrantClient(":memory:"), FakeOpenSearch()
    argv = [
        str(root),
        "--profile",
        "ventes",
        "--profiles-dir",
        str(ROOT / "configs" / "profiles"),
        "--hardware",
        "cpu",
        "--hardware-dir",
        str(ROOT / "configs" / "hardware"),
        "--models",
        str(ROOT / "configs" / "models.yaml"),
        "--store",
        str(tmp_path / "data" / "ventes.db"),
        "--interval",
        "0.01",
    ]
    code = main(
        argv,
        max_ticks=2,
        embedder_factory=lambda spec, hardware: FakeEmbedder(),
        qdrant=qdrant,
        opensearch=opensearch,
    )
    assert code == 0
    assert "Surveillance de" in capsys.readouterr().out
    [collection] = [c.name for c in qdrant.get_collections().collections]
    vectors = QdrantIndex(qdrant, collection, FakeEmbedder().dimension)
    assert vectors.count() == OpenSearchIndex(opensearch, collection).count() > 0
    WatcherLock(tmp_path / "data" / "ventes.db").acquire().release()  # verrou libéré
