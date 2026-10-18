"""Réponse sourcée : recherche, rédaction, gardes, verdict, trace, avec un LLM
scripté (aucun appel à Ollama) sur le corpus « ventes » indexé en mémoire."""

import sys
from pathlib import Path

import pytest

from luciole.core.hardware import load_hardware_profile
from luciole.core.profile import RetrievalSpec, load_profile
from luciole.generation.answer import Answerer, decide, format_passages
from luciole.generation.ask import main as ask_main
from luciole.generation.guards import GuardReport
from luciole.generation.llm import Completion
from luciole.generation.prompts import load_prompts
from luciole.generation.trace import JsonlTracer
from luciole.retrieval.hybrid import HybridRetriever
from luciole.store.sqlite import Condition

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_index_pipeline import Setup  # noqa: E402
from test_retrieval import WordReranker  # noqa: E402

VENTES = load_profile(ROOT / "configs" / "profiles" / "ventes")
PROMPTS = load_prompts(VENTES.path)
CPU = load_hardware_profile(ROOT / "configs" / "hardware" / "cpu.yaml")
QUESTION = "Pourquoi l'atelier trouve douze mille euros au-dessus de ce qu'on peut engager ?"
GOOD = (
    "Paul Dubois juge que douze mille euros la première année dépassent ce que l'atelier "
    "peut engager cette année [Source: reunion-3.transcript.md]. Une embauche est prévue à "
    "l'automne et reste la priorité. Le dossier est remis à janvier."
)
EVASIVE = "Les extraits ne mentionnent pas cette information."
LOOP = "\n".join(["- L'atelier hésite sur le budget de cette année encore."] * 9)


class ScriptedLLM:
    """Rend les réponses données, dans l'ordre, et note chaque appel."""

    model = "llm-scripte"

    def __init__(self, *texts, truncated=()):
        self.texts = list(texts)
        self.truncated = set(truncated)
        self.calls: list[dict] = []

    def generate(self, system, prompt, *, options=None, schema=None):
        i = len(self.calls)
        self.calls.append({"system": system, "prompt": prompt, "options": dict(options or {})})
        return Completion(
            self.texts[min(i, len(self.texts) - 1)],
            self.model,
            truncated=i in self.truncated,
            prompt_tokens=100,
            completion_tokens=50,
        )


class ListTracer:
    def __init__(self):
        self.traces = []

    def write(self, trace):
        self.traces.append(trace)


class ScaledReranker(WordReranker):
    def score(self, query, passages):
        return [s / 100 for s in super().score(query, passages)]


@pytest.fixture()
def env(tmp_path):
    s = Setup(tmp_path)
    s.pipeline.run(s.root)
    yield s
    s.store.close()


def answerer(env, llm, hardware=CPU, profile=None, tracer=None):
    profile = profile or VENTES.profile
    spec = profile.retrieval
    retriever = HybridRetriever(
        spec, env.embedder, env.vectors, env.lexical, env.store, ScaledReranker()
    )
    return Answerer(profile, PROMPTS, retriever, llm, hardware, tracer=tracer)


def test_reponse_propre_suffisante(env):
    llm, tracer = ScriptedLLM(GOOD), ListTracer()
    result = answerer(env, llm, tracer=tracer).answer(QUESTION)
    assert result.verdict == "SUFFISANT" and result.guards.clean
    assert result.answer == GOOD and 0 < result.confidence <= 1
    call = llm.calls[0]
    assert call["system"] == PROMPTS.system
    assert "[Source: atelier-dubois/reunion-3.transcript.md, p. 1]" in call["prompt"]
    assert QUESTION in call["prompt"]
    gen = VENTES.profile.generation
    assert call["options"] == {
        "temperature": gen.temperature,
        "seed": gen.seed,
        "num_predict": gen.max_tokens,
        **gen.options,
    }
    body = result.to_dict()
    assert set(body) == {
        "answer",
        "citations",
        "verdict",
        "confidence",
        "trace_id",
        "sources_used",
        "guards",
    }
    assert "atelier-dubois/reunion-3.transcript.md" in body["sources_used"]
    assert all("text" not in c for c in body["citations"])
    [trace] = tracer.traces
    assert trace["trace_id"] == result.trace_id and trace["verdict"] == "SUFFISANT"
    assert trace["models"] == {
        "embedding": "faux-embedder",
        "reranker": "faux-reranker",
        "llm": "llm-scripte",
    }
    assert trace["versions"]["app"] and len(trace["searches"]) == 1
    assert trace["generation"][0]["step"] == "rédaction"


def test_esquive_puis_passe_elargie(env):
    llm = ScriptedLLM(EVASIVE, GOOD)
    result = answerer(env, llm).answer(QUESTION)
    assert result.verdict == "SUFFISANT" and result.answer == GOOD
    steps = [s["step"] for s in result.trace["generation"]]
    assert steps == ["rédaction", "passe élargie"]
    assert result.trace["searches"][1]["deep"] is True


def test_esquive_deux_fois_insuffisant(env):
    llm = ScriptedLLM(EVASIVE, EVASIVE)
    result = answerer(env, llm).answer(QUESTION)
    assert result.verdict == "INSUFFISANT" and result.confidence == 0.0
    assert result.guards.evasive and result.answer == EVASIVE and len(llm.calls) == 2


def test_esquive_sans_relance_si_le_profil_le_dit(env):
    profile = VENTES.profile.model_copy(deep=True)
    profile.generation.guards.retry_on_evasion = False
    llm = ScriptedLLM(EVASIVE, GOOD)
    assert answerer(env, llm, profile=profile).answer(QUESTION).verdict == "INSUFFISANT"
    assert len(llm.calls) == 1


def test_boucle_relancee_sous_penalite(env):
    llm = ScriptedLLM(LOOP, GOOD)
    result = answerer(env, llm).answer(QUESTION)
    assert result.verdict == "SUFFISANT" and result.answer == GOOD
    retry = VENTES.profile.generation.retry_options
    assert all(llm.calls[1]["options"][k] == v for k, v in retry.items())
    assert [s["step"] for s in result.trace["generation"]] == ["rédaction", "relance"]


def test_tronquee_deux_fois_partiel(env):
    llm = ScriptedLLM(GOOD, GOOD, truncated={0, 1})
    result = answerer(env, llm).answer(QUESTION)
    assert result.verdict == "PARTIEL" and result.guards.truncated


def test_nombre_sans_extrait_signale(env):
    text = GOOD + " Le devis prévoyait 27 500 euros."
    result = answerer(env, ScriptedLLM(text)).answer(QUESTION)
    assert result.verdict == "PARTIEL" and result.guards.numbers_outside == ["27 500"]
    assert result.answer.endswith("_Nombres sans extrait correspondant dans le lot : 27 500._")


def test_aucun_passage_sans_appel_au_llm(env):
    llm = ScriptedLLM(GOOD)
    result = answerer(env, llm).answer(
        QUESTION, conditions=[Condition("dossier.issue", "eq", "abandonne")]
    )
    assert result.verdict == "INSUFFISANT" and result.answer == PROMPTS.no_results.strip()
    assert llm.calls == [] and result.passages == []


def test_fenetre_du_llm(env):
    small = CPU.model_copy(update={"llm_num_ctx": 2048})
    with pytest.raises(ValueError, match="fenêtre"):
        answerer(env, ScriptedLLM(GOOD), hardware=small)
    spec = RetrievalSpec(rerank_top_k=10, rerank_candidates=10)
    retriever = HybridRetriever(
        spec, env.embedder, env.vectors, env.lexical, env.store, ScaledReranker()
    )
    passages = retriever.retrieve(QUESTION).passages
    costs = [p.unit.token_count for p in passages]
    text, used = format_passages(passages, costs[0] + costs[1])
    assert len(used) == 2 and text.count("[Source: ") == 2
    _, first = format_passages(passages, 1)
    assert len(first) == 1  # le premier passe toujours


def test_verdict():
    assert decide(GuardReport(), True) == "SUFFISANT"
    assert decide(GuardReport(), False) == "INSUFFISANT"
    assert decide(GuardReport(evasive=True), True) == "INSUFFISANT"
    assert decide(GuardReport(pages_outside=["p. 3"]), True) == "PARTIEL"


def test_traces_jsonl(tmp_path):
    tracer = JsonlTracer(tmp_path / "traces")
    path = tracer.write({"trace_id": "abc", "question": "é"})
    tracer.write({"trace_id": "def"})
    assert path.read_text(encoding="utf-8").count("\n") == 2
    assert tracer.find("abc") == {"trace_id": "abc", "question": "é"}
    assert tracer.find("zzz") is None
    assert JsonlTracer(tmp_path / "rien").find("abc") is None


def test_ligne_de_commande(env, tmp_path, capsys):
    tracer = JsonlTracer(tmp_path / "traces")
    code = ask_main(
        [QUESTION, "--profile", "ventes", "--hardware", "cpu", "--store", str(env.store.path)],
        llm_factory=lambda hw, gen: ScriptedLLM(GOOD),
        tracer=tracer,
        embedder_factory=lambda spec, hw: env.embedder,
        reranker_factory=lambda spec, hw: ScaledReranker(),
        qdrant=env.qdrant,
        opensearch=env.opensearch,
    )
    out = capsys.readouterr().out
    assert code == 0 and GOOD in out and "Verdict : SUFFISANT" in out
    assert "reunion-3.transcript.md" in out and "Trace : " in out
    trace_id = out.rsplit("Trace : ", 1)[1].strip()
    assert tracer.find(trace_id)["verdict"] == "SUFFISANT"
