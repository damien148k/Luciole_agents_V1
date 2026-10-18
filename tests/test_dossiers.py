"""Découverte des dossiers, identifiants de documents et métadonnées par règles,
sur le corpus fictif « ventes » et sur des cas construits."""

import shutil
from pathlib import Path

import pytest

from luciole.core.dossier import content_hash, make_document_id, normalize_relative_path
from luciole.core.profile import FieldSpec, UseCaseProfile, load_profile
from luciole.ingestion.dossiers import (
    SOURCE_DOCUMENT,
    SOURCE_DOCUMENTS,
    SOURCE_DOSSIER_FILE,
    SOURCE_EXTRACTOR,
    SOURCE_FILENAME,
    filename_values,
    format_report,
    main,
    scan,
)
from luciole.ingestion.registry import default_registry

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "corpus" / "ventes"
VENTES = load_profile(ROOT / "configs" / "profiles" / "ventes").profile
MRAE = load_profile(ROOT / "configs" / "profiles" / "mrae").profile


@pytest.fixture(scope="module")
def corpus_scan():
    return scan(CORPUS, VENTES, default_registry())


# ---------------------------------------------------------------------------
# Identifiants
# ---------------------------------------------------------------------------
def test_chemin_relatif_canonique():
    assert normalize_relative_path("a\\b\\c.pdf") == "a/b/c.pdf"
    assert normalize_relative_path("./a//b.pdf") == "a/b.pdf"
    assert normalize_relative_path("Réunion.md") == "Réunion.md", "NFD → NFC"
    with pytest.raises(ValueError):
        normalize_relative_path("../hors.pdf")


def test_document_id_depend_du_chemin_et_du_contenu():
    a = make_document_id("dossier-a/Tome 4.pdf", "h1")
    assert a == make_document_id("dossier-a\\Tome 4.pdf", "h1"), "même fichier sous Windows"
    assert a != make_document_id("dossier-b/Tome 4.pdf", "h1"), "même nom, autre dossier"
    assert a != make_document_id("dossier-a/Tome 4.pdf", "h2"), "contenu modifié"
    assert len(a) == 32


def test_content_hash(tmp_path):
    f = tmp_path / "x.txt"
    f.write_bytes(b"abc")
    assert content_hash(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


# ---------------------------------------------------------------------------
# Corpus « ventes » : porte de l'étape 2 (côté métadonnées)
# ---------------------------------------------------------------------------
def test_deux_fichiers_de_meme_nom_coexistent(corpus_scan):
    """Porte de l'étape 2 : deux « reunion-1.transcript.md » dans deux dossiers."""
    firsts = [d for d in corpus_scan.documents if d.file_name == "reunion-1.transcript.md"]
    assert {d.dossier_id for d in firsts} == {"atelier-dubois", "cabinet-roche"}
    assert len({d.document_id for d in firsts}) == 2
    assert len({d.document_id for d in corpus_scan.documents}) == len(corpus_scan.documents) == 5


def test_champs_d_une_reunion_fictive_extraits_et_valides(corpus_scan):
    """Porte de l'étape 2 : les champs d'une réunion sont extraits et validés."""
    doc = next(
        d
        for d in corpus_scan.documents
        if d.relative_path == "atelier-dubois/reunion-2.transcript.md"
    )
    assert doc.valid and doc.issues == []
    assert doc.metadata == {
        "numero_reunion": 2,
        "date": "2026-03-04",
        "type_document": "transcription",
    }
    assert set(doc.sources.values()) == {SOURCE_DOCUMENT}
    assert doc.doc_type == "transcript" and doc.page_count == 1


def test_numero_lu_dans_le_nom_et_date_francaise(corpus_scan):
    doc = next(
        d
        for d in corpus_scan.documents
        if d.relative_path == "cabinet-roche/reunion-1.transcript.md"
    )
    assert doc.metadata["numero_reunion"] == 1 and doc.sources["numero_reunion"] == SOURCE_FILENAME
    assert doc.metadata["date"] == "2026-02-12", "12/02/2026 ramené au format ISO"
    assert doc.valid


def test_dossier_depuis_dossier_yaml(corpus_scan):
    roche = next(d for d in corpus_scan.dossiers if d.dossier_id == "cabinet-roche")
    assert roche.metadata == {
        "client": "Cabinet Roche (fictif)",
        "produit": "Luciole Pro",
        "commercial": "Julien Bernard",
        "issue": "gagne",
    }, "« Gagné » ramené à la valeur déclarée"
    assert set(roche.sources.values()) == {SOURCE_DOSSIER_FILE}
    assert roche.valid and roche.issues == []


def test_dossier_consolide_depuis_les_reunions_la_derniere_fait_foi(corpus_scan):
    dubois = next(d for d in corpus_scan.dossiers if d.dossier_id == "atelier-dubois")
    assert dubois.metadata["issue"] == "perdu", "réunion 3 : perdu, malgré « en cours » avant"
    assert dubois.metadata["commercial"] == "Camille Martin"
    assert set(dubois.sources.values()) == {SOURCE_DOCUMENTS}
    assert dubois.valid
    [warning] = dubois.issues
    assert warning.level == "warning" and warning.field == "issue"
    assert "reunion-3" in warning.message


def test_rapport_lisible(corpus_scan):
    text = format_report(corpus_scan)
    assert "OK Dossier « atelier-dubois » : 3 document(s)" in text
    assert "issue = gagne   [fichier_dossier]" in text


# ---------------------------------------------------------------------------
# Cas construits
# ---------------------------------------------------------------------------
def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_fichiers_ecartes_et_illisibles(tmp_path):
    _write(tmp_path / "a-la-racine.md", "# perdu")
    _write(
        tmp_path / "client-x" / "reunion-1.transcript.md",
        "---\ndate: 2026-01-05\n---\n[00:01] A : bonjour",
    )
    _write(tmp_path / "client-x" / "photo.jpg", "binaire")
    _write(tmp_path / "client-x" / "~$verrou.docx", "verrou Office")
    _write(tmp_path / "client-x" / ".cache" / "x.md", "caché")
    (tmp_path / "client-x" / "abime.docx").write_bytes(b"pas un docx")
    result = scan(tmp_path, VENTES, default_registry())
    reasons = {s.relative_path: s.reason for s in result.skipped}
    assert reasons["a-la-racine.md"].startswith("fichier hors de tout dossier")
    assert reasons["client-x/photo.jpg"] == "format non pris en charge"
    assert reasons["client-x/abime.docx"].startswith("lecture impossible")
    assert set(reasons) == {"a-la-racine.md", "client-x/photo.jpg", "client-x/abime.docx"}
    assert [d.relative_path for d in result.documents] == ["client-x/reunion-1.transcript.md"]


def test_champs_requis_manquants_signales_sans_bloquer(tmp_path):
    _write(tmp_path / "client-y" / "notes.md", "# Notes\n\nSans date ni numéro.")
    _write(
        tmp_path / "client-y" / "dossier.yaml", "client: Y\nproduit: P\nissue: signé\nbudget: 3\n"
    )
    result = scan(tmp_path, VENTES, default_registry())
    [doc] = result.documents
    assert not doc.valid
    assert {i.field for i in doc.issues if i.level == "error"} == {"numero_reunion", "date"}
    [dossier] = result.dossiers
    errors = {(i.field, i.message.split(" ")[0]) for i in dossier.issues if i.level == "error"}
    assert errors == {("issue", "valeur"), ("issue", "champ"), ("commercial", "champ")}
    assert any(i.field == "budget" and i.level == "warning" for i in dossier.issues)


def test_conflit_sans_ordre_est_une_erreur(tmp_path):
    profile = UseCaseProfile.model_validate(
        {
            "name": "t",
            "title": "T",
            "entities": {"dossier": {"porteur": {"type": "string"}}},
        }
    )
    _write(tmp_path / "p" / "a.md", "---\nporteur: Alpha\n---\ntexte")
    _write(tmp_path / "p" / "b.md", "---\nporteur: Beta\n---\ntexte")
    [dossier] = scan(tmp_path, profile, default_registry()).dossiers
    assert "porteur" not in dossier.metadata
    assert [i.level for i in dossier.issues] == ["error"]
    assert "aucun `document_order`" in dossier.issues[0].message


def test_dossier_yaml_tranche_un_conflit(tmp_path):
    shutil.copytree(CORPUS / "atelier-dubois", tmp_path / "atelier-dubois")
    _write(tmp_path / "atelier-dubois" / "dossier.yaml", "issue: abandonné\n")
    [dossier] = scan(tmp_path, VENTES, default_registry()).dossiers
    assert dossier.metadata["issue"] == "abandonne"
    assert dossier.sources["issue"] == SOURCE_DOSSIER_FILE
    assert dossier.issues == [], "pas d'avertissement : dossier.yaml fait foi"


class FakeExtractor:
    """Extracteur factice : renvoie des valeurs fixes et note les champs demandés."""

    name = "faux-llm"

    def __init__(self, values: dict) -> None:
        self.values = values
        self.asked: list[list[str]] = []

    def extract(self, document, fields: dict[str, FieldSpec]) -> dict:
        self.asked.append(sorted(fields))
        return self.values


def test_extracteur_ne_comble_que_les_champs_manquants(tmp_path):
    _write(tmp_path / "client-z" / "reunion-2.md", "# Compte rendu\n\nRéunion du 3 mars.")
    extractor = FakeExtractor({"date": "03/03/2026", "numero_reunion": 99, "type_document": "fax"})
    [doc] = scan(tmp_path, VENTES, default_registry(), extractor=extractor).documents
    assert extractor.asked == [["date", "type_document"]], "le numéro vient déjà du nom"
    assert doc.metadata["numero_reunion"] == 2 and doc.sources["numero_reunion"] == SOURCE_FILENAME
    assert doc.metadata["date"] == "2026-03-03" and doc.sources["date"] == SOURCE_EXTRACTOR
    assert "type_document" not in doc.metadata, "valeur hors liste refusée, pas corrigée"
    assert [(i.field, i.level) for i in doc.issues] == [("type_document", "error")]


def test_extracteur_en_echec_ne_bloque_pas(tmp_path):
    class Broken:
        name = "en-panne"

        def extract(self, document, fields):
            raise RuntimeError("modèle indisponible")

    _write(tmp_path / "c" / "reunion-1.md", "texte")
    [doc] = scan(tmp_path, VENTES, default_registry(), extractor=Broken()).documents
    assert doc.metadata == {"numero_reunion": 1}
    assert any("modèle indisponible" in i.message for i in doc.issues)


def test_on_document_recoit_chaque_document_une_fois(corpus_scan):
    seen = []
    scan(
        CORPUS,
        VENTES,
        default_registry(),
        on_document=lambda rec, parsed: seen.append(parsed.file_name),
    )
    assert sorted(seen) == sorted(d.file_name for d in corpus_scan.documents)


def test_motif_mrae_sur_des_noms_reels():
    assert filename_values(MRAE, "02_Projet_EIE_tome_1_volet_projet.pdf") == {"tome": "1"}
    assert filename_values(MRAE, "Tome 4 - Milieu naturel.pdf") == {"tome": "4"}
    assert filename_values(MRAE, "Resume_non_technique.pdf") == {}
    assert filename_values(VENTES, "Réunion 3 - CR.docx") == {"numero_reunion": "3"}


def test_ligne_de_commande(tmp_path, capsys):
    store = tmp_path / "ventes.db"
    assert (
        main(
            [
                str(CORPUS),
                "--profile",
                "ventes",
                "--profiles-dir",
                str(ROOT / "configs" / "profiles"),
                "--store",
                str(store),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Dossier « cabinet-roche » : 2 document(s)" in out
    assert "5 nouveau(x)" in out and store.is_file()
