"""Conversion et validation des métadonnées contre le schéma du profil."""

import datetime as dt

import pytest

from luciole.core.metadata import coerce_value, merge_sources
from luciole.core.profile import FieldSpec

TEXT = FieldSpec(type="string")
INT = FieldSpec(type="integer")
NUM = FieldSpec(type="number")
DATE = FieldSpec(type="date")
BOOL = FieldSpec(type="boolean")
ISSUE = FieldSpec(type="enum", values=["en_cours", "gagne", "perdu", "abandonne"])


@pytest.mark.parametrize(
    ("spec", "raw", "expected"),
    [
        (TEXT, "  Camille   Martin ", "Camille Martin"),
        (TEXT, 42, "42"),
        (INT, 3, 3),
        (INT, "3", 3),
        (INT, " 12 ", 12),
        (INT, 2.0, 2),
        (NUM, "3,5", 3.5),
        (NUM, 7, 7.0),
        (DATE, "2026-02-10", "2026-02-10"),
        (DATE, "12/02/2026", "2026-02-12"),
        (DATE, "12.02.2026", "2026-02-12"),
        (DATE, dt.date(2026, 3, 4), "2026-03-04"),
        (DATE, dt.datetime(2026, 3, 4, 9, 30), "2026-03-04"),
        (BOOL, "oui", True),
        (BOOL, "Non", False),
        (BOOL, 1, True),
        (ISSUE, "Gagné", "gagne"),
        (ISSUE, "EN COURS", "en_cours"),
        (ISSUE, "en-cours", "en_cours"),
        (ISSUE, "abandonné", "abandonne"),
    ],
)
def test_conversion_canonique(spec, raw, expected):
    assert coerce_value(spec, raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_valeur_vide_est_absente(raw):
    assert coerce_value(INT, raw) is None


@pytest.mark.parametrize(
    ("spec", "raw", "message"),
    [
        (INT, "trois", "entier attendu"),
        (INT, 2.5, "entier attendu"),
        (INT, True, "entier attendu"),
        (NUM, "beaucoup", "nombre attendu"),
        (DATE, "mars 2026", "date attendue"),
        (DATE, "31/02/2026", "date attendue"),
        (BOOL, "peut-être", "oui ou non"),
        (ISSUE, "signé", "hors de la liste"),
        (TEXT, ["a"], "texte attendu"),
    ],
)
def test_valeur_invalide_jamais_devinee(spec, raw, message):
    with pytest.raises(ValueError, match=message):
        coerce_value(spec, raw)


SCHEMA = {
    "numero_reunion": FieldSpec(type="integer", required=True),
    "date": FieldSpec(type="date", required=True),
    "type_document": FieldSpec(type="enum", values=["transcription", "email"]),
}


def test_priorite_des_sources_et_cles_hors_schema_ignorees():
    result = merge_sources(
        SCHEMA,
        [
            ("document", {"numero_reunion": 2, "speaker_count": 3, "date": "2026-03-04"}),
            ("nom_de_fichier", {"numero_reunion": "9"}),
        ],
    )
    assert result.values == {"numero_reunion": 2, "date": "2026-03-04"}
    assert result.sources == {"numero_reunion": "document", "date": "document"}
    assert result.issues == [] and result.missing == []


def test_source_suivante_quand_la_premiere_est_vide():
    result = merge_sources(
        SCHEMA, [("document", {"numero_reunion": ""}), ("nom", {"numero_reunion": "4"})]
    )
    assert result.values["numero_reunion"] == 4 and result.sources["numero_reunion"] == "nom"


def test_valeur_invalide_comblee_devient_avertissement():
    result = merge_sources(
        SCHEMA,
        [
            ("document", {"numero_reunion": "deux", "date": "2026-03-04"}),
            ("nom", {"numero_reunion": "2"}),
        ],
    )
    assert result.values["numero_reunion"] == 2
    assert [(i.level, i.field, i.source) for i in result.issues] == [
        ("warning", "numero_reunion", "document")
    ]
    assert result.errors == []


def test_champ_requis_absent_et_valeur_invalide_sont_des_erreurs():
    result = merge_sources(SCHEMA, [("document", {"date": "hier", "type_document": "fax"})])
    assert result.values == {}
    assert result.missing == ["numero_reunion", "date"]
    assert sorted((i.field, i.message.split(" ")[0]) for i in result.errors) == [
        ("date", "champ"),
        ("date", "date"),
        ("numero_reunion", "champ"),
        ("type_document", "valeur"),
    ]
