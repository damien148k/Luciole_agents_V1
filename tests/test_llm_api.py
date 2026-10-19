"""Client Ollama (transport httpx simulé), sortie typée, prompts du profil, et
API HTTP (``POST /api/rag/query``, santé, traces) avec un LLM scripté."""

import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from luciole.api.app import create_app
from luciole.generation.llm import OllamaLLM, extract
from luciole.generation.prompts import PromptSet, load_prompts
from luciole.generation.trace import JsonlTracer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_agent import MRAE_LOOP, VENTES_LOOP  # noqa: E402
from test_answer import GOOD, QUESTION, ScriptedLLM, answerer  # noqa: E402
from test_index_pipeline import Setup  # noqa: E402


# --- Ollama ---------------------------------------------------------------
def ollama(handler) -> OllamaLLM:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaLLM(
        "qwen2.5:14b", num_ctx=16384, base_options={"top_k": 20}, url="http://x", client=client
    )


def test_appel_ollama():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5:14b",
                "message": {"content": "Réponse."},
                "done_reason": "stop",
                "prompt_eval_count": 812,
                "eval_count": 40,
            },
        )

    out = ollama(handler).generate("système", "demande", options={"seed": 42, "temperature": 0})
    body = seen["body"]
    assert seen["url"] == "http://x/api/chat" and body["stream"] is False
    assert body["messages"] == [
        {"role": "system", "content": "système"},
        {"role": "user", "content": "demande"},
    ]
    assert body["options"] == {"top_k": 20, "num_ctx": 16384, "seed": 42, "temperature": 0}
    assert "format" not in body
    assert out.text == "Réponse." and not out.truncated
    assert (out.prompt_tokens, out.completion_tokens) == (812, 40)


def test_reponse_tronquee_et_erreurs():
    cut = ollama(
        lambda r: httpx.Response(200, json={"message": {"content": "x"}, "done_reason": "length"})
    )
    assert cut.generate("s", "p").truncated
    refused = ollama(lambda r: httpx.Response(404, text="model not found"))
    with pytest.raises(RuntimeError, match="404"):
        refused.generate("s", "p")

    def down(request):
        raise httpx.ConnectError("refusé")

    with pytest.raises(RuntimeError, match="ollama pull qwen2.5:14b"):
        ollama(down).generate("s", "p")


class Champs(BaseModel):
    commercial: str
    numero_reunion: int


def test_sortie_typee_validee():
    seen = {}

    def handler(request):
        seen["format"] = json.loads(request.content)["format"]
        return httpx.Response(
            200, json={"message": {"content": '{"commercial": "Camille", "numero_reunion": 3}'}}
        )

    out = extract(ollama(handler), "s", "p", Champs)
    assert out == Champs(commercial="Camille", numero_reunion=3)
    assert seen["format"]["properties"]["numero_reunion"]["type"] == "integer"
    bad = ollama(lambda r: httpx.Response(200, json={"message": {"content": '{"commercial": 1}'}}))
    with pytest.raises(ValueError, match="non conforme"):
        extract(bad, "s", "p", Champs)
    junk = ollama(lambda r: httpx.Response(200, json={"message": {"content": "pas du JSON"}}))
    with pytest.raises(ValueError, match="non conforme"):
        extract(junk, "s", "p", Champs)


# --- prompts --------------------------------------------------------------
def test_prompts_des_profils():
    for name in ("mrae", "ventes"):
        prompts = load_prompts(ROOT / "configs" / "profiles" / name)
        rendered = prompts.render_answer("Q?", "[Source: a.pdf, p. 1]\nTexte")
        assert "Q?" in rendered and "[Source: a.pdf, p. 1]" in rendered
    assert "MRAe" in load_prompts(ROOT / "configs" / "profiles" / "mrae").system


def test_gabarits_controles():
    base = {"system": "s", "no_results": "rien"}
    PromptSet(**base, answer="{passages} {question} {{littéral}}")
    with pytest.raises(ValidationError, match="inconnu"):
        PromptSet(**base, answer="{passages} {question} {objet}")
    with pytest.raises(ValidationError, match="absent"):
        PromptSet(**base, answer="{question}")
    with pytest.raises(ValidationError, match="illisible"):
        PromptSet(**base, answer="{passages} {question} {")
    with pytest.raises(FileNotFoundError):
        load_prompts(ROOT / "configs")


# --- API ------------------------------------------------------------------
@pytest.fixture()
def api(tmp_path):
    """Le magasin s'ouvre dans le fil de travail de l'API, comme en production
    (une connexion SQLite reste dans le fil qui l'a ouverte)."""
    tracer = JsonlTracer(tmp_path / "traces")
    built = []

    def factory(use_case):
        built.append(use_case)
        s = Setup(tmp_path)
        s.pipeline.run(s.root)
        return answerer(s, ScriptedLLM(GOOD), tracer=tracer)

    client = TestClient(
        create_app(
            factory,
            tracer=tracer,
            use_cases=lambda: ["mrae", "ventes"],
            loops=lambda use_case: MRAE_LOOP if use_case == "mrae" else VENTES_LOOP,
        )
    )
    return client, built


def test_rag_query(api):
    client, built = api
    r = client.post("/api/rag/query", json={"question": QUESTION, "use_case": "ventes"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] == "SUFFISANT" and body["answer"] == GOOD
    assert body["citations"][0]["citation"].startswith("atelier-dubois/reunion-3.transcript.md")
    assert set(body["citations"][0]) >= {"chunk_id", "document_id", "page_labels", "cited"}
    trace = client.get(f"/api/traces/{body['trace_id']}")
    assert trace.status_code == 200 and trace.json()["question"] == QUESTION
    client.post("/api/rag/query", json={"question": QUESTION, "use_case": "ventes"})
    assert built == ["ventes"]  # chargé une fois
    health = client.get("/api/health").json()
    assert health["status"] == "ok" and health["versions"]["app"]
    assert health["loaded"]["ventes"]["llm"] == "llm-scripte"


def test_rag_query_filtres_et_erreurs(api):
    client, _ = api
    r = client.post(
        "/api/rag/query",
        json={
            "question": QUESTION,
            "use_case": "ventes",
            "filters": [{"field": "dossier.issue", "op": "eq", "value": "abandonne"}],
        },
    )
    assert r.status_code == 200 and r.json()["verdict"] == "INSUFFISANT"
    assert r.json()["citations"] == []
    assert (
        client.post("/api/rag/query", json={"question": "x", "use_case": "inconnu"}).status_code
        == 404
    )
    assert (
        client.post("/api/rag/query", json={"question": "", "use_case": "ventes"}).status_code
        == 422
    )
    assert (
        client.post("/api/rag/query", json={"question": "  ", "use_case": "ventes"}).status_code
        == 422
    )
    assert (
        client.post(
            "/api/rag/query", json={"question": "x", "use_case": "ventes", "autre": 1}
        ).status_code
        == 422
    )
    assert client.get("/api/traces/inconnue").status_code == 404


def test_agent_query(api):
    client, built = api
    r = client.post("/api/agent/query", json={"question": QUESTION, "use_case": "mrae"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] == "SUFFISANT" and body["answer"] == GOOD
    assert body["stopped_reason"] == "reponse" and body["steps_used"] == 3
    assert body["path"] == ["rechercher", "rediger", "verdict", "reponse"]
    assert body["matched_rules"] == []
    trace = client.get(f"/api/traces/{body['trace_id']}").json()
    assert trace["kind"] == "agent" and trace["path"] == body["path"]
    rag = client.post("/api/rag/query", json={"question": QUESTION, "use_case": "mrae"})
    assert rag.json()["answer"] == body["answer"] and built == ["mrae"]


def test_agent_query_boucle_pas_encore_executable(api):
    client, _ = api
    r = client.post("/api/agent/query", json={"question": QUESTION, "use_case": "ventes"})
    assert r.status_code == 422 and "pas encore exécutable" in r.json()["detail"]
