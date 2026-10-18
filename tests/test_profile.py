"""Profils de cas d'usage : les profils livrés sont valides, et un YAML
incohérent est refusé plutôt que toléré (règle du post-mortem Belacom)."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from luciole.core.profile import Loop, UseCaseProfile, list_profiles, load_profile

PROFILES = Path(__file__).resolve().parents[1] / "configs" / "profiles"


def test_les_profils_livres_sont_listes():
    assert list_profiles(PROFILES) == ["mrae", "ventes"]


@pytest.mark.parametrize("nom", ["mrae", "ventes"])
def test_chaque_profil_livre_se_charge(nom):
    bundle = load_profile(PROFILES / nom)
    assert bundle.profile.name == nom
    assert bundle.loop.name == nom
    assert bundle.loop.terminal_nodes(), "une boucle a au moins un nœud terminal"


def test_le_profil_ventes_decrit_une_suite_de_reunions():
    bundle = load_profile(PROFILES / "ventes")
    doc = bundle.profile.entities.document
    assert doc["numero_reunion"].type == "integer" and doc["numero_reunion"].required
    assert set(bundle.profile.entities.dossier["issue"].values) >= {"gagne", "perdu"}


def test_profil_absent():
    with pytest.raises(FileNotFoundError):
        load_profile(PROFILES / "inexistant")


def test_cle_inconnue_refusee():
    with pytest.raises(ValidationError):
        UseCaseProfile.model_validate({"name": "x", "title": "X", "routing_rules": []})


def test_enum_sans_valeurs_refuse():
    with pytest.raises(ValidationError):
        UseCaseProfile.model_validate(
            {"name": "x", "title": "X", "entities": {"document": {"etat": {"type": "enum"}}}}
        )


def _loop(nodes: dict, start: str = "a") -> dict:
    return {"name": "t", "start": start, "nodes": nodes}


def test_boucle_sans_terminal_refusee():
    with pytest.raises(ValidationError, match="terminal"):
        Loop.model_validate(_loop({"a": {"type": "retrieve", "default": "a"}}))


def test_boucle_vers_noeud_inexistant_refusee():
    with pytest.raises(ValidationError, match="n'existe pas"):
        Loop.model_validate(
            _loop({"a": {"type": "retrieve", "default": "zz"}, "fin": {"type": "terminal"}})
        )


def test_noeud_terminal_non_reentrant():
    with pytest.raises(ValidationError, match="réentrant"):
        Loop.model_validate(
            _loop(
                {
                    "a": {"type": "retrieve", "default": "fin"},
                    "fin": {"type": "terminal", "default": "a"},
                }
            )
        )


def test_outil_inconnu_refuse():
    with pytest.raises(ValidationError):
        Loop.model_validate(
            _loop(
                {"a": {"type": "search_documents", "default": "fin"}, "fin": {"type": "terminal"}}
            )
        )


def test_noeud_outil_sans_sortie_refuse():
    with pytest.raises(ValidationError, match="default"):
        Loop.model_validate(_loop({"a": {"type": "retrieve"}, "fin": {"type": "terminal"}}))


def test_nom_du_profil_doit_etre_celui_du_dossier(tmp_path):
    d = tmp_path / "autre"
    d.mkdir()
    (d / "profile.yaml").write_text(
        yaml.safe_dump({"name": "mrae", "title": "X"}), encoding="utf-8"
    )
    (d / "loop.yaml").write_text(
        yaml.safe_dump(
            _loop({"a": {"type": "verdict", "default": "fin"}, "fin": {"type": "terminal"}})
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="dossier"):
        load_profile(d)


# ---------------------------------------------------------------------------
# Règles d'ingestion (étape 2)
# ---------------------------------------------------------------------------
def _profile(ingestion: dict, document: dict | None = None, dossier: dict | None = None) -> dict:
    return {
        "name": "x",
        "title": "X",
        "entities": {
            "document": document if document is not None else {"numero": {"type": "integer"}},
            "dossier": dossier or {},
        },
        "ingestion": ingestion,
    }


def test_motif_de_nom_de_fichier_valide():
    p = UseCaseProfile.model_validate(_profile({"filename_patterns": [r"reunion-(?P<numero>\d+)"]}))
    assert p.ingestion.compiled_patterns()[0].search("REUNION-4.md").group("numero") == "4"


@pytest.mark.parametrize(
    ("ingestion", "message"),
    [
        ({"filename_patterns": ["(?P<numero>"]}, "expression régulière invalide"),
        ({"filename_patterns": [r"reunion-\d+"]}, "aucun groupe nommé"),
        ({"filename_patterns": [r"(?P<tome>\d+)"]}, "n'est pas un champ de document"),
        ({"document_order": "date"}, "n'est pas un champ de document"),
        ({"dossier_file": "../dossier.yaml"}, "dossier_file"),
    ],
)
def test_regles_d_ingestion_incoherentes_refusees(ingestion, message):
    with pytest.raises(ValidationError, match=message):
        UseCaseProfile.model_validate(_profile(ingestion))


def test_ordre_sur_un_champ_texte_refuse():
    with pytest.raises(ValidationError, match="nombre ou une date"):
        UseCaseProfile.model_validate(
            _profile({"document_order": "titre"}, document={"titre": {"type": "string"}})
        )


def test_champ_a_la_fois_dossier_et_document_refuse():
    with pytest.raises(ValidationError, match="à la fois"):
        UseCaseProfile.model_validate(
            _profile({}, document={"date": {"type": "date"}}, dossier={"date": {"type": "date"}})
        )
