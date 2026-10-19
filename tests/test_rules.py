"""Règles des boucles : grammaire à liste blanche, validée au chargement et
réellement évaluée (post-mortem Belacom : des règles YAML jamais exécutées)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from luciole.agents.rules import STATE_FIELDS, RuleError, parse
from luciole.core.profile import Loop, load_profile

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("source", "state", "expected"),
    [
        ("verdict == 'INSUFFISANT'", {"verdict": "INSUFFISANT"}, True),
        ("verdict == 'INSUFFISANT'", {"verdict": "PARTIEL"}, False),
        (
            "verdict != 'SUFFISANT' and refinements_left > 0",
            {"verdict": "PARTIEL", "refinements_left": 1},
            True,
        ),
        (
            "verdict != 'SUFFISANT' and refinements_left > 0",
            {"verdict": "PARTIEL", "refinements_left": 0},
            False,
        ),
        ("count == 0 or evasive", {"count": 3, "evasive": True}, True),
        ("not evasive", {"evasive": False}, True),
        ("verdict in ('PARTIEL', 'INSUFFISANT')", {"verdict": "PARTIEL"}, True),
        ("verdict not in ['SUFFISANT']", {"verdict": "SUFFISANT"}, False),
        ("0.2 <= confidence < 0.7", {"confidence": 0.5}, True),
        ("0.2 <= confidence < 0.7", {"confidence": 0.9}, False),
        ("evasive == true", {"evasive": True}, True),
        ("(count == 0) and (step >= 2)", {"count": 0, "step": 2}, True),
    ],
)
def test_evaluation(source, state, expected):
    assert parse(source)(state) is expected


def test_valeur_absente_jamais_une_erreur():
    """Avant la première rédaction, verdict et confiance valent None."""
    assert parse("confidence < 0.5")({}) is False
    assert parse("confidence >= 0.5")({}) is False
    assert parse("verdict == 'SUFFISANT'")({}) is False
    assert parse("verdict != 'SUFFISANT'")({}) is True


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os').system('echo')",
        "verdict.upper() == 'X'",
        "len(verdict) > 0",
        "count + 1 > 2",
        "metadata['severity'] == 'critical'",
        "severity == 'critical'",
        "lambda: True",
        "[x for x in (1, 2)]",
        "count == (1 + 1, 2)",
        "verdict is None",
        "verdict ==",
    ],
)
def test_hors_grammaire_refuse(source):
    with pytest.raises(RuleError):
        parse(source)


def test_champs_lus():
    assert parse("verdict == 'X' and refinements_left > 0").fields == {
        "verdict",
        "refinements_left",
    }
    assert set(STATE_FIELDS) >= {"verdict", "confidence", "count", "refinements_left"}


def test_profil_refuse_une_regle_hors_grammaire():
    loop = {
        "name": "t",
        "start": "a",
        "nodes": {
            "a": {
                "type": "verdict",
                "rules": [{"when": "severity == 1", "then": "fin"}],
                "default": "fin",
            },
            "fin": {"type": "terminal"},
        },
    }
    with pytest.raises(ValidationError, match="champ inconnu"):
        Loop.model_validate(loop)


def test_on_budget_doit_etre_terminal():
    loop = {
        "name": "t",
        "start": "a",
        "on_budget": "a",
        "nodes": {"a": {"type": "verdict", "default": "fin"}, "fin": {"type": "terminal"}},
    }
    with pytest.raises(ValidationError, match="on_budget"):
        Loop.model_validate(loop)


@pytest.mark.parametrize("case", ["mrae", "ventes"])
def test_regles_des_profils_valides(case):
    loop = load_profile(ROOT / "configs" / "profiles" / case).loop
    for node in loop.nodes.values():
        for rule in node.rules:
            parse(rule.when)
