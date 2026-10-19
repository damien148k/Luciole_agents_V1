"""Moteur de boucles avec la boucle « RAG en nœuds » de l'étape 5a, un LLM
scripté et le corpus « ventes » indexé en mémoire. La boucle MRAe complète
(analyse, questions composées) est vérifiée dans ``test_agent_mrae.py``.

Ce qui est vérifié : le code choisit la suite et l'arrêt (règles, budget,
raffinement borné), la meilleure réponse est gardée, la trace dit tout."""

import sys
from pathlib import Path

import pytest
import yaml

from luciole.agents.engine import BUDGET_EXHAUSTED, LoopAgent
from luciole.agents.tools import build_agent
from luciole.core.profile import Loop, load_profile
from luciole.store.sqlite import Condition

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_answer import (  # noqa: E402
    EVASIVE,
    GOOD,
    PROMPTS,
    QUESTION,
    ListTracer,
    ScriptedLLM,
    answerer,
    env,  # noqa: F401
)

#: La boucle MRAe de l'étape 5a : le RAG de l'étape 4 en nœuds, une passe élargie.
RAG_LOOP = Loop.model_validate(
    yaml.safe_load(
        """
name: mrae
start: rechercher
max_steps: 8
max_refinements: 1
on_budget: reponse
nodes:
  rechercher:
    type: retrieve
    rules: [{when: "count == 0", then: sans_reponse}]
    default: rediger
  rediger: {type: synthesize, default: verdict}
  verdict:
    type: verdict
    rules:
      - {when: "verdict != 'SUFFISANT' and refinements_left > 0", then: elargir}
      - {when: "verdict == 'INSUFFISANT'", then: sans_reponse}
    default: reponse
  elargir:
    type: retrieve
    params: {mode: deep, refine: true}
    rules: [{when: "count == 0", then: verdict}]
    default: rediger
  reponse: {type: terminal}
  sans_reponse: {type: terminal}
"""
    )
)
VENTES_LOOP = load_profile(ROOT / "configs" / "profiles" / "ventes").loop
PARTIAL = GOOD + " Le devis prévoyait 27 500 euros."


def agent(env, llm, loop=RAG_LOOP, tracer=None):  # noqa: F811
    return build_agent(loop, answerer(env, llm), tracer=tracer)


def test_reponse_directe(env):  # noqa: F811
    llm, tracer = ScriptedLLM(GOOD), ListTracer()
    result = agent(env, llm, tracer=tracer).run(QUESTION)
    assert result.stopped_reason == "reponse" and result.answer.verdict == "SUFFISANT"
    assert result.path == ["rechercher", "rediger", "verdict", "reponse"]
    assert result.steps_used == 3 and result.matched_rules == [] and len(llm.calls) == 1
    assert result.answer.answer == GOOD
    body = result.to_dict()
    assert body["stopped_reason"] == "reponse" and body["trace_id"] == result.trace_id
    assert "atelier-dubois/reunion-3.transcript.md" in body["sources_used"]
    [trace] = tracer.traces
    assert trace["kind"] == "agent" and trace["loop"] == "mrae"
    assert trace["trace_id"] == result.trace_id == result.answer.trace_id
    assert [s["node"] for s in trace["steps"]] == ["rechercher", "rediger", "verdict"]
    assert trace["steps"][2]["state"]["verdict"] == "SUFFISANT"
    assert trace["models"]["llm"] == "llm-scripte" and trace["profile"] == "ventes"


def test_meme_reponse_que_le_rag_seul(env):  # noqa: F811
    """Sur une bonne réponse du premier coup, la boucle rend ce que rend le RAG."""
    rag = answerer(env, ScriptedLLM(GOOD)).answer(QUESTION)
    loop = agent(env, ScriptedLLM(GOOD)).answer(QUESTION)
    assert (loop.answer, loop.verdict, loop.confidence) == (rag.answer, rag.verdict, rag.confidence)
    assert [p.unit.chunk_id for p in loop.passages] == [p.unit.chunk_id for p in rag.passages]


def test_esquive_puis_passe_elargie(env):  # noqa: F811
    llm = ScriptedLLM(EVASIVE, GOOD)
    result = agent(env, llm).run(QUESTION)
    assert result.answer.verdict == "SUFFISANT" and result.answer.answer == GOOD
    assert result.path == [
        "rechercher",
        "rediger",
        "verdict",
        "elargir",
        "rediger",
        "verdict",
        "reponse",
    ]
    [rule] = result.matched_rules
    assert rule["then"] == "elargir" and "refinements_left > 0" in rule["when"]
    assert [s["deep"] for s in result.trace["searches"]] == [False, True]
    assert len(llm.calls) == 2


def test_raffinement_borne_en_code(env):  # noqa: F811
    """Deux esquives : une seule passe élargie, puis arrêt sans réponse."""
    llm = ScriptedLLM(EVASIVE, EVASIVE, GOOD)
    result = agent(env, llm).run(QUESTION)
    assert result.stopped_reason == "sans_reponse" and len(llm.calls) == 2
    assert result.answer.verdict == "INSUFFISANT" and result.answer.guards.evasive
    assert [r["then"] for r in result.matched_rules] == ["elargir", "sans_reponse"]


def test_jamais_moins_bien_apres_la_passe_elargie(env):  # noqa: F811
    """Réponse PARTIEL, passe élargie qui esquive : la réponse PARTIEL reste."""
    llm = ScriptedLLM(PARTIAL, EVASIVE)
    result = agent(env, llm).run(QUESTION)
    assert result.stopped_reason == "reponse" and len(llm.calls) == 2
    assert result.answer.verdict == "PARTIEL" and result.answer.guards.numbers_outside == ["27 500"]
    verdicts = [s["detail"] for s in result.trace["steps"] if s["type"] == "verdict"]
    assert verdicts == [
        {"latest": "PARTIEL", "kept": True},
        {"latest": "INSUFFISANT", "kept": False},
    ]


def test_aucun_passage_sans_appel_au_llm(env):  # noqa: F811
    llm = ScriptedLLM(GOOD)
    result = agent(env, llm).run(
        QUESTION, conditions=[Condition("dossier.issue", "eq", "abandonne")]
    )
    assert result.stopped_reason == "sans_reponse" and llm.calls == []
    assert result.path == ["rechercher", "sans_reponse"]
    assert result.answer.answer == PROMPTS.no_results.strip()
    assert result.answer.verdict == "INSUFFISANT"


def test_plafond_de_noeuds(env):  # noqa: F811
    tight = RAG_LOOP.model_copy(update={"max_steps": 2})
    result = agent(env, ScriptedLLM(GOOD), loop=tight).run(QUESTION)
    assert result.stopped_reason == "reponse" and result.steps_used == 2
    assert result.path == ["rechercher", "rediger", "reponse"]
    assert result.trace["steps"][-1] == {"node": "verdict", "stopped": "max_steps"}
    # Le verdict n'a pas tourné : la dernière rédaction est rendue telle quelle.
    assert result.answer.answer == GOOD
    no_budget = tight.model_copy(update={"on_budget": None})
    assert agent(env, ScriptedLLM(GOOD), loop=no_budget).run(QUESTION).stopped_reason == (
        BUDGET_EXHAUSTED
    )


def test_noeud_de_raffinement_sans_budget(env):  # noqa: F811
    """Une règle qui renvoie vers un raffinement épuisé ne fait pas tourner l'outil."""
    loop = Loop.model_validate(
        {
            "name": "t",
            "start": "chercher",
            "max_refinements": 0,
            "on_budget": "fin",
            "nodes": {
                "chercher": {"type": "retrieve", "default": "encore"},
                "encore": {"type": "retrieve", "params": {"refine": True}, "default": "fin"},
                "fin": {"type": "terminal"},
            },
        }
    )
    result = agent(env, ScriptedLLM(GOOD), loop=loop).run(QUESTION)
    assert result.path == ["chercher", "fin"] and result.steps_used == 1
    assert result.trace["steps"][-1] == {"node": "encore", "stopped": "max_refinements"}
    assert len(result.trace["searches"]) == 1


def test_outil_inconnu_refuse(env):  # noqa: F811
    with pytest.raises(ValueError, match="pas encore exécutable"):
        agent(env, ScriptedLLM(GOOD), loop=VENTES_LOOP)


def test_mode_de_recherche_inconnu(env):  # noqa: F811
    loop = Loop.model_validate(
        {
            "name": "t",
            "start": "chercher",
            "nodes": {
                "chercher": {"type": "retrieve", "params": {"mode": "tout"}, "default": "fin"},
                "fin": {"type": "terminal"},
            },
        }
    )
    with pytest.raises(ValueError, match="mode"):
        agent(env, ScriptedLLM(GOOD), loop=loop).run(QUESTION)


def test_regles_dans_l_ordre(env):  # noqa: F811
    """La première règle qui matche gagne, même si une suivante matche aussi."""
    loop = Loop.model_validate(
        {
            "name": "t",
            "start": "chercher",
            "nodes": {
                "chercher": {
                    "type": "retrieve",
                    "rules": [
                        {"when": "count > 0", "then": "premiere"},
                        {"when": "count > 1", "then": "seconde"},
                    ],
                    "default": "seconde",
                },
                "premiere": {"type": "terminal"},
                "seconde": {"type": "terminal"},
            },
        }
    )
    result = agent(env, ScriptedLLM(GOOD), loop=loop).run(QUESTION)
    assert result.stopped_reason == "premiere"
    assert isinstance(agent(env, ScriptedLLM(GOOD), loop=loop), LoopAgent)
