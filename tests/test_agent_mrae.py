"""Boucle MRAe de l'étape 5b (``configs/profiles/mrae/loop-questions.yaml`` et
``prompts.yaml``) : analyse de la remarque en JSON validé, questions de
recherche composées par le code, recherche par question, rédaction cadrée.

LLM scripté (le premier appel rend l'analyse JSON) et corpus « ventes »
indexé en mémoire : seuls les prompts et la boucle sont ceux de la MRAe."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

from luciole.agents.engine import State
from luciole.agents.tools import ComposeTool, build_agent, check_references, interleave
from luciole.core.profile import Loop
from luciole.generation.answer import Answerer
from luciole.generation.prompts import PromptSet, load_prompts
from luciole.retrieval.hybrid import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_answer import (  # noqa: E402
    CPU,
    EVASIVE,
    GOOD,
    QUESTION,
    VENTES,
    ScaledReranker,
    ScriptedLLM,
    answerer,
    env,  # noqa: F401
)

MRAE_DIR = ROOT / "configs" / "profiles" / "mrae"
MRAE_LOOP = Loop.model_validate(
    yaml.safe_load((MRAE_DIR / "loop-questions.yaml").read_text(encoding="utf-8"))
)
MRAE_PROMPTS = load_prompts(MRAE_DIR)
ANALYSIS = {
    "objet": "budget de l'atelier",
    "type_objet": "etude",
    "acte": "préciser le montant engageable",
    "aspects": ["montant de douze mille euros la première année"],
    "motifs": [],
    "demontrer": "Le dossier doit établir ce que l'atelier peut engager cette année.",
}
ANALYSIS_JSON = json.dumps(ANALYSIS, ensure_ascii=False)


def mrae_answerer(env, llm):  # noqa: F811
    profile = VENTES.profile
    retriever = HybridRetriever(
        profile.retrieval, env.embedder, env.vectors, env.lexical, env.store, ScaledReranker()
    )
    return Answerer(profile, MRAE_PROMPTS, retriever, llm, CPU)


def agent(env, llm):  # noqa: F811
    return build_agent(MRAE_LOOP, mrae_answerer(env, llm))


def test_analyse_questions_puis_reponse(env):  # noqa: F811
    llm = ScriptedLLM(ANALYSIS_JSON, GOOD)
    result = agent(env, llm).run(QUESTION)
    assert result.stopped_reason == "reponse" and result.answer.answer == GOOD
    assert result.path == [
        "analyser",
        "composer",
        "rechercher_questions",
        "rediger",
        "verdict",
        "reponse",
    ]
    steps = {s["node"]: s["detail"] for s in result.trace["steps"]}
    assert steps["analyser"] == {"schema": "analyse_remarque", "values": ANALYSIS}
    questions = steps["composer"]["questions"]
    assert len(questions) == 2
    assert questions[0].startswith("Que présente le dossier concernant « budget de l'atelier »")
    assert "montant de douze mille euros la première année" in questions[1]
    # La remarque puis chaque question : trois recherches, résultats entrelacés.
    assert [s["query"] for s in result.trace["searches"]] == [QUESTION, *questions]
    assert steps["rechercher_questions"]["queries"] == 3
    extract_call, draft_call = llm.calls
    assert QUESTION in extract_call["prompt"] and '"objet"' in extract_call["prompt"]
    assert "ANALYSE DE LA REMARQUE" in draft_call["prompt"]
    assert "objet visé : budget de l'atelier" in draft_call["prompt"]
    assert "thèmes invoqués : aucun" in draft_call["prompt"]


def test_le_premier_passage_du_rag_reste_en_tete(env):  # noqa: F811
    """Les passages de la remarque elle-même gardent leur rang 1."""
    rag = answerer(env, ScriptedLLM(GOOD)).answer(QUESTION)
    loop = agent(env, ScriptedLLM(ANALYSIS_JSON, GOOD)).answer(QUESTION)
    assert loop.passages[0].unit.chunk_id == rag.passages[0].unit.chunk_id
    rag_ids = {p.unit.chunk_id for p in rag.passages}
    assert rag_ids <= {p.unit.chunk_id for p in loop.passages}


@pytest.mark.parametrize(
    "bad",
    [
        "pas du JSON",
        json.dumps({**ANALYSIS, "type_objet": "autre"}),
        json.dumps({k: v for k, v in ANALYSIS.items() if k != "aspects"}),
        json.dumps({**ANALYSIS, "items": []}),
    ],
)
def test_analyse_invalide_recherche_directe(env, bad):  # noqa: F811
    """Une analyse non conforme n'est ni devinée ni réparée : repli sur la
    recherche de la seule remarque, sans cadrage."""
    llm = ScriptedLLM(bad, GOOD)
    result = agent(env, llm).run(QUESTION)
    assert result.stopped_reason == "reponse" and result.answer.answer == GOOD
    assert result.path == ["analyser", "rechercher", "rediger", "verdict", "reponse"]
    [rule, *_] = result.matched_rules
    assert rule["node"] == "analyser" and rule["then"] == "rechercher"
    assert "non conforme" in result.trace["steps"][0]["detail"]["error"]
    assert len(result.trace["searches"]) == 1
    assert "ANALYSE DE LA REMARQUE" not in llm.calls[1]["prompt"]


def test_esquive_passe_elargie_par_question(env):  # noqa: F811
    llm = ScriptedLLM(ANALYSIS_JSON, EVASIVE, GOOD)
    result = agent(env, llm).run(QUESTION)
    assert result.answer.verdict == "SUFFISANT" and result.answer.answer == GOOD
    assert result.path[-4:] == ["elargir", "rediger", "verdict", "reponse"]
    assert [s["deep"] for s in result.trace["searches"]] == [False] * 3 + [True] * 3
    assert len(llm.calls) == 3  # une analyse, deux rédactions


def compose(values, **params):
    state = State(QUESTION, extracted={"analyse_remarque": values})
    tool = ComposeTool(SimpleNamespace(prompts=MRAE_PROMPTS))
    defaults = {
        "from": "analyse_remarque",
        "first": "objet_{type_objet}",
        "fallback": "objet_etude",
        "each": {"aspects": "aspect", "motifs": "motif"},
    }
    tool.run(defaults | params, state)
    return state


def test_questions_bornees_les_motifs_cedent_en_premier():
    values = {**ANALYSIS, "aspects": ["a1", "a2", "a3"], "motifs": ["m1", "m2", "m3", "m4"]}
    state = compose(values)
    assert state.count == 6 == len(state.questions)
    assert sum("a" in q[-6:] for q in state.questions[1:4]) == 3
    assert ["m1" in state.questions[4], "m2" in state.questions[5]] == [True, True]
    state = compose({**values, "aspects": [f"a{i}" for i in range(9)]}, max_questions=4)
    assert len(state.questions) == 4 and not any("m1" in q for q in state.questions)


def test_question_d_objet_selon_le_type():
    for kind, words in [
        ("comparaison", "options étudiées"),
        ("mesure", "calendrier de mise en œuvre"),
        ("element", "modalités de réalisation"),
        ("forme", "cartes ou figures"),
    ]:
        state = compose({**ANALYSIS, "type_objet": kind, "aspects": []})
        assert state.questions == [state.questions[0]] and words in state.questions[0]
    state = compose(ANALYSIS, first="inexistant_{type_objet}")
    assert "méthode et protocole" in state.questions[0]  # repli objet_etude


def test_aspects_vides_ignores_et_extraction_absente():
    state = compose({**ANALYSIS, "aspects": ["", "  "], "motifs": ["paysage"]})
    assert len(state.questions) == 2 and "« paysage »" in state.questions[1]
    with pytest.raises(ValueError, match="aucune extraction"):
        ComposeTool(SimpleNamespace(prompts=MRAE_PROMPTS)).run(
            {"from": "analyse_remarque"}, State(QUESTION)
        )


def passage(chunk_id, file_name="etude.pdf"):
    return SimpleNamespace(unit=SimpleNamespace(chunk_id=chunk_id, file_name=file_name))


def test_entrelacement_sans_doublon():
    a, b, c, d = (passage(x) for x in "abcd")
    merged, capped = interleave([[a, b, c], [b, d], [a]])
    assert [p.unit.chunk_id for p in merged] == ["a", "b", "d", "c"] and capped == 0
    assert interleave([]) == ([], 0)


def test_documents_secondaires_bornes_sur_l_ensemble():
    """Deux passages du résumé non technique au plus, toutes questions confondues."""
    rnt = [passage(f"r{i}", "RNT_projet.pdf") for i in range(4)]
    tome = [passage(f"t{i}") for i in range(2)]
    merged, capped = interleave(
        [[rnt[0], tome[0]], [rnt[1], rnt[2]], [rnt[3], tome[1]]], ["RNT"], 2
    )
    assert [p.unit.chunk_id for p in merged] == ["r0", "r1", "t0", "t1"] and capped == 2


def test_references_verifiees_au_chargement(env):  # noqa: F811
    """Une boucle qui nomme un schéma ou un gabarit absent est refusée tout de suite."""
    with pytest.raises(ValueError, match="schéma « analyse_remarque » absent"):
        build_agent(MRAE_LOOP, answerer(env, ScriptedLLM(GOOD)))  # prompts « ventes »
    check_references(MRAE_LOOP, MRAE_PROMPTS)
    no_framing = MRAE_PROMPTS.model_copy(
        update={"answer": MRAE_PROMPTS.answer.replace("{framing}", "")}
    )
    with pytest.raises(ValueError, match="framing"):
        check_references(MRAE_LOOP, no_framing)


def test_prompts_mrae_valides():
    spec = MRAE_PROMPTS.extract["analyse_remarque"]
    model = spec.model()
    assert model.model_validate(ANALYSIS).objet == "budget de l'atelier"
    with pytest.raises(ValidationError):
        model.model_validate({**ANALYSIS, "type_objet": "autre"})
    assert set(MRAE_PROMPTS.questions) == {
        "objet_comparaison",
        "objet_etude",
        "objet_mesure",
        "objet_element",
        "objet_forme",
        "aspect",
        "motif",
    }
    assert MRAE_PROMPTS.render_framing("analyse_remarque", ANALYSIS).startswith(
        "ANALYSE DE LA REMARQUE"
    )


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"questions": {"x": "Que dit {inconnu} ?"}}, "question « x »"),
        ({"framing": {"analyse_remarque": "objet : {sujet}"}}, "cadrage"),
        (
            {"extract": {"a": {"system": "s", "prompt": "{question} {autre}", "fields": {}}}},
            "extract",
        ),
        (
            {"extract": {"a": {"system": "s", "prompt": "{question}", "fields": {"t": []}}}},
            "énumération vide",
        ),
    ],
)
def test_prompts_refuses_au_chargement(update, match):
    base = MRAE_PROMPTS.model_dump()
    with pytest.raises(ValidationError, match=match):
        PromptSet.model_validate(base | update)
