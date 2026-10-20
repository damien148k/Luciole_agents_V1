"""Boucle « ventes » (``configs/profiles/ventes/loop.yaml``) sur les 10 dossiers
fictifs d'``evaluation/ventes`` indexés en mémoire, avec un LLM factice :
une extraction (appel avec schéma) rend des signaux JSON, une rédaction rend
le texte donné.

Ce qui est vérifié : le code retrouve le dossier et lit ses réunions dans
l'ordre, compare aux dossiers du même produit, et la rédaction reçoit
l'issue du dossier et les signaux de chaque réunion."""

import json
import sys
from pathlib import Path

import pytest

from luciole.agents.records import normalize
from luciole.agents.tools import build_agent, check_references
from luciole.core.profile import load_profile
from luciole.generation.answer import Answerer
from luciole.generation.llm import Completion
from luciole.generation.prompts import load_prompts
from luciole.retrieval.hybrid import HybridRetriever
from luciole.store.sqlite import Condition

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_answer import CPU, ListTracer, ScaledReranker  # noqa: E402
from test_index_pipeline import Setup  # noqa: E402

VENTES = load_profile(ROOT / "configs" / "profiles" / "ventes")
PROMPTS = load_prompts(VENTES.path)
DOSSIERS = ROOT / "evaluation" / "ventes" / "dossiers"
SIGNALS = {
    "objections": ["le prix est trop élevé"],
    "engagements": ["envoyer une proposition"],
    "decideurs": ["Henri Girard"],
    "prochaine_etape": "une démonstration",
    "tonalite": "neutre",
}
ANALYSIS = "Le décideur n'a jamais vu la démonstration (notaires-girard/reunion-3.transcript.md)."


class FakeLLM:
    """Signaux JSON quand un schéma est demandé, sinon le texte de rédaction ;
    ``bad`` : rangs (0, 1…) des extractions qui rendent un JSON invalide."""

    model = "llm-factice"

    def __init__(self, text=ANALYSIS, signals=SIGNALS, bad=()):
        self.text, self.signals, self.bad = text, signals, set(bad)
        self.calls: list[dict] = []
        self.extractions = 0

    def generate(self, system, prompt, *, options=None, schema=None):
        self.calls.append({"prompt": prompt, "schema": schema is not None})
        if schema is None:
            out = self.text
        else:
            out = "pas du JSON" if self.extractions in self.bad else json.dumps(self.signals)
            self.extractions += 1
        return Completion(out, self.model, prompt_tokens=100, completion_tokens=50)


@pytest.fixture(scope="module")
def dossiers(tmp_path_factory):
    s = Setup(tmp_path_factory.mktemp("ventes"), corpus=DOSSIERS)
    s.pipeline.run(s.root)
    yield s
    s.store.close()


def agent(s, llm, loop=VENTES.loop, tracer=None):
    retriever = HybridRetriever(
        VENTES.profile.retrieval, s.embedder, s.vectors, s.lexical, s.store, ScaledReranker()
    )
    answerer = Answerer(VENTES.profile, PROMPTS, retriever, llm, CPU)
    return build_agent(loop, answerer, tracer=tracer)


def test_dossier_lu_compare_et_explique(dossiers):
    llm, tracer = FakeLLM(), ListTracer()
    result = agent(dossiers, llm, tracer=tracer).run(
        "Pourquoi le dossier de l'Office notarial Girard a-t-il été perdu ?"
    )
    assert result.stopped_reason == "analyse" and result.answer.answer == ANALYSIS
    assert result.path == [
        "selectionner",
        "signaux",
        "comparables",
        "comparer",
        "expliquer",
        "verdict",
        "analyse",
    ]
    steps = {s["node"]: s["detail"] for s in result.trace["steps"]}
    meetings = [f"notaires-girard/reunion-{i}.transcript.md" for i in (1, 2, 3)]
    assert steps["selectionner"]["dossier"] == "notaires-girard"
    assert steps["selectionner"]["documents"] == meetings
    # Une extraction par réunion, dans l'ordre, sur le texte entier de la réunion.
    extractions = [c for c in llm.calls if c["schema"]]
    assert len(extractions) == 3 and len(llm.calls) == 4
    lines = ["Qu'est-ce qui vous prend", "Mon père n'a pas pu venir", "nous avons choisi"]
    for call, path, line in zip(extractions, meetings, lines, strict=True):
        assert path in call["prompt"] and line in call["prompt"]
    assert [v["rang"] for v in steps["signaux"]["values"]] == [1, 2, 3]
    # Les comparables : même produit (Luciole Pro), gagnés ou perdus, sans le dossier.
    assert sorted(steps["comparables"]["comparables"]) == [
        "agence-architecture-fontaine",
        "bureau-etudes-lambert",
        "geometres-blanc",
        "industries-perrin",
    ]
    assert steps["comparables"]["counts"] == {"gagne": 2, "perdu": 2}
    assert steps["comparer"]["queries"] == ["le prix est trop élevé"]
    assert steps["comparer"]["passages"]
    assert not any(p.startswith("notaires-girard/") for p in steps["comparer"]["passages"])
    # Les réunions du dossier d'abord, dans l'ordre, puis les extraits comparés.
    used = [p.unit.relative_path for p in result.answer.passages]
    assert used[:3] == meetings and len(used) > 3
    prompt = llm.calls[-1]["prompt"]
    assert "DOSSIER ANALYSÉ : notaires-girard (Office notarial Girard (fictif)" in prompt
    assert "Issue du dossier à ce jour : perdu." in prompt
    assert f"- {meetings[2]} : objections le prix est trop élevé" in prompt
    assert "DOSSIERS COMPARABLES (produit = Luciole Pro) : gagne : 2 ; perdu : 2." in prompt
    assert prompt.index("DOSSIER ANALYSÉ") < prompt.index("EXTRAITS DES RÉUNIONS")
    [trace] = tracer.traces
    assert trace["kind"] == "agent" and trace["loop"] == "ventes"


def test_dossier_donne_par_les_conditions(dossiers):
    """L'API ou le banc donnent le dossier : la demande n'a pas à le nommer."""
    llm = FakeLLM()
    result = agent(dossiers, llm).run(
        "Pourquoi ont-ils signé si vite ?",
        conditions=[Condition("dossier_id", "eq", "syndic-mercier")],
    )
    steps = {s["node"]: s["detail"] for s in result.trace["steps"]}
    assert steps["selectionner"]["dossier"] == "syndic-mercier"
    assert len([c for c in llm.calls if c["schema"]]) == 2
    assert "Issue du dossier à ce jour : gagne." in llm.calls[-1]["prompt"]


@pytest.mark.parametrize(
    "question",
    [
        "Pourquoi le dossier Dupont a-t-il été perdu ?",
        "Compare le Syndic Mercier et Industries Perrin.",
    ],
)
def test_aucun_ou_plusieurs_dossiers_rien_n_est_devine(dossiers, question):
    llm = FakeLLM()
    result = agent(dossiers, llm).run(question)
    assert result.stopped_reason == "dossier_inconnu" and llm.calls == []
    assert result.path == ["selectionner", "dossier_inconnu"]
    assert result.answer.answer == PROMPTS.no_results.strip()


def test_extraction_invalide_laissee_de_cote(dossiers):
    llm = FakeLLM(bad={1})
    result = agent(dossiers, llm).run("Pourquoi Industries Perrin a refusé notre offre ?")
    steps = {s["node"]: s["detail"] for s in result.trace["steps"]}
    assert [v["rang"] for v in steps["signaux"]["values"]] == [1, 3]
    [error] = steps["signaux"]["errors"]
    assert error["source"] == "industries-perrin/reunion-2.transcript.md"
    assert "non conforme" in error["error"]
    assert "industries-perrin/reunion-2.transcript.md : objections" not in llm.calls[-1]["prompt"]
    assert result.stopped_reason == "analyse"


def test_reunion_trop_longue_signalee_pas_tronquee(dossiers):
    llm = FakeLLM()
    built = agent(dossiers, llm)
    built.tools["extract"].answerer.budget = 10
    result = built.run("Pourquoi le Syndic Mercier a-t-il signé si vite ?")
    steps = {s["node"]: s["detail"] for s in result.trace["steps"]}
    assert steps["signaux"]["values"] == []
    assert all("trop long" in e["error"] for e in steps["signaux"]["errors"])
    assert not any(c["schema"] for c in llm.calls)


def test_sans_comparable_pas_de_comparaison(dossiers):
    loop = VENTES.loop.model_copy(deep=True)
    loop.nodes["comparables"].params["values"] = ["en_cours"]  # aucun en Essentiel
    result = agent(dossiers, FakeLLM(), loop=loop).run(
        "Pourquoi le Syndic Mercier a-t-il signé si vite ?"
    )
    assert "comparer" not in result.path and result.stopped_reason == "analyse"
    assert result.matched_rules[0]["then"] == "expliquer"


def test_cadrage_qui_lit_un_champ_absent_refuse():
    prompts = PROMPTS.model_copy(
        update={"framing": PROMPTS.framing | {"dossier": "Budget : {budget}"}}
    )
    with pytest.raises(ValueError, match=r"cadrage « dossier » : champ\(s\) inconnu\(s\)"):
        check_references(VENTES.loop, prompts, set(VENTES.profile.entities.dossier))
    check_references(VENTES.loop, PROMPTS, set(VENTES.profile.entities.dossier))


def test_nom_normalise():
    assert normalize("Géomètres Blanc et associés (fictif)") == "geometres blanc et associes"
    assert normalize("l'Office notarial Girard ?") == "l office notarial girard"
