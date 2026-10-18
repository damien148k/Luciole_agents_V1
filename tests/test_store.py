"""Magasin structuré SQLite : synchronisation sans doublon, filtres, comptages."""

import shutil
from pathlib import Path

import pytest

from luciole.core.profile import load_profile
from luciole.ingestion.dossiers import scan
from luciole.ingestion.registry import default_registry
from luciole.store.sqlite import Condition, MetadataStore

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "corpus" / "ventes"
VENTES = load_profile(ROOT / "configs" / "profiles" / "ventes").profile
MRAE = load_profile(ROOT / "configs" / "profiles" / "mrae").profile


@pytest.fixture()
def root(tmp_path) -> Path:
    target = tmp_path / "ventes"
    shutil.copytree(CORPUS, target)
    return target


@pytest.fixture()
def store(tmp_path, root):
    s = MetadataStore(tmp_path / "ventes.db", VENTES)
    s.sync(scan(root, VENTES, default_registry()))
    yield s
    s.close()


def test_resynchronisation_sans_doublon(store, root):
    report = store.sync(scan(root, VENTES, default_registry()))
    assert len(report.unchanged) == 5 and not (report.new or report.replaced or report.removed)
    assert len(store.select("document")) == 5


def test_fichier_modifie_remplace_sa_fiche(store, root):
    path = root / "cabinet-roche" / "reunion-2.transcript.md"
    old = next(
        d
        for d in store.select("document")
        if d.relative_path.endswith("roche/reunion-2.transcript.md")
    )
    path.write_text(
        path.read_text(encoding="utf-8") + "\n[00:05:00] Marc Faure : Merci.\n", encoding="utf-8"
    )
    report = store.sync(scan(root, VENTES, default_registry()))
    [(new_id, old_id)] = report.replaced.items()
    assert old_id == old.document_id and new_id != old_id
    assert store.get_document(old_id) is None and store.get_document(new_id) is not None
    assert len(store.select("document")) == 5


def test_fichier_et_dossier_supprimes(store, root):
    (root / "atelier-dubois" / "reunion-3.transcript.md").unlink()
    shutil.rmtree(root / "cabinet-roche")
    report = store.sync(scan(root, VENTES, default_registry()))
    assert len(report.removed) == 3 and report.removed_dossiers == ["cabinet-roche"]
    assert [d.dossier_id for d in store.select("dossier")] == ["atelier-dubois"]
    assert len(store.select("document")) == 2


def test_relecture_d_une_fiche(store):
    dossier = store.get_dossier("atelier-dubois")
    assert dossier.metadata["issue"] == "perdu" and dossier.sources["issue"] == "documents"
    assert [i.field for i in dossier.issues] == ["issue"]
    assert store.get_dossier("inexistant") is None


def test_filtre_documents_par_champ_du_dossier_et_ordre_chronologique(store):
    """« Les réunions des dossiers perdus », rangées par numéro de réunion."""
    docs = store.select("document", [Condition("dossier.issue", "eq", "Perdu")])
    assert [(d.dossier_id, d.metadata["numero_reunion"]) for d in docs] == [
        ("atelier-dubois", 1),
        ("atelier-dubois", 2),
        ("atelier-dubois", 3),
    ]


def test_filtres_combines_dates_et_listes(store):
    docs = store.select(
        "document",
        [
            Condition("date", "ge", "01/03/2026"),
            Condition("dossier.commercial", "in", ["Camille Martin"]),
        ],
    )
    assert [d.metadata["date"] for d in docs] == ["2026-03-04", "2026-03-25"]
    assert store.select("document", [Condition("numero_reunion", "in", [])]) == []
    assert [d.dossier_id for d in store.select("dossier", [Condition("issue", "ne", "perdu")])] == [
        "cabinet-roche"
    ]
    assert len(store.select("document", [Condition("doc_type", "eq", "transcript")])) == 5


def test_comptages(store):
    assert store.count_by("dossier", "issue") == {"gagne": 1, "perdu": 1}
    assert store.count_by("document", "dossier.commercial") == {
        "Camille Martin": 3,
        "Julien Bernard": 2,
    }
    assert store.count_by(
        "document", "numero_reunion", [Condition("dossier.issue", "eq", "gagné")]
    ) == {
        1: 1,
        2: 1,
    }


@pytest.mark.parametrize(
    "name",
    ["budget", "dossier.inconnu", "issue) OR 1=1 --", "dossier.issue'; DROP TABLE documents; --"],
)
def test_champ_inconnu_ou_malveillant_refuse(store, name):
    with pytest.raises(ValueError, match="champ"):
        store.select("document", [Condition(name, "eq", "x")])
    assert len(store.select("document")) == 5, "rien n'a été exécuté"


def test_valeur_de_filtre_validee(store):
    with pytest.raises(ValueError, match="hors de la liste"):
        store.select("dossier", [Condition("issue", "eq", "signé")])


def test_magasin_d_un_autre_cas_d_usage_refuse(tmp_path, store):
    with pytest.raises(ValueError, match="ventes"):
        MetadataStore(tmp_path / "ventes.db", MRAE)


def test_magasin_en_memoire():
    with MetadataStore(":memory:", MRAE) as s:
        assert s.select("dossier") == [] and s.count_by("dossier", "projet") == {}
