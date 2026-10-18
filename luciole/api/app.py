"""API HTTP : ``POST /api/rag/query``, ``GET /api/health``, ``GET /api/traces/{id}``.

Le contrat de ``/api/rag/query`` est celui de ``rag-ingestion-spec.md`` (7) :
la boucle de l'étape 5 appellera ce RAG comme un outil, sans connaître son
fonctionnement. Un cas d'usage est chargé à sa première question (modèles,
index, magasin), puis gardé.

Les questions passent une à une par un seul fil de travail : le GPU ne sert
qu'une réponse à la fois, et le magasin SQLite reste dans le fil qui l'a
ouvert (une connexion partagée entre fils concurrents a fait tomber V5, voir
le post-mortem). ``/api/health`` répond pendant qu'une question est en cours.

    uv run --extra embed uvicorn luciole.api.app:app --port 8000
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from luciole.core.versions import Versions
from luciole.generation.answer import Answerer
from luciole.generation.trace import JsonlTracer
from luciole.store.sqlite import Condition


class Filter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1, description="champ de document, ou dossier.<champ>")
    op: Literal["eq", "ne", "lt", "le", "gt", "ge", "in"] = "eq"
    value: object


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    use_case: str = Field(min_length=1, description="nom du profil (mrae, ventes…)")
    filters: list[Filter] = Field(default_factory=list)
    document_ids: list[str] | None = None


class Citation(BaseModel):
    citation: str
    chunk_id: str
    document_id: str
    dossier_id: str
    relative_path: str
    page_start: int | None
    page_end: int | None
    page_labels: list[str]
    cited: bool


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    verdict: Literal["SUFFISANT", "PARTIEL", "INSUFFISANT"]
    confidence: float
    trace_id: str
    sources_used: list[str]
    guards: dict


def create_app(
    answerer_factory: Callable[[str], Answerer],
    *,
    tracer: JsonlTracer | None = None,
    use_cases: Callable[[], list[str]] | None = None,
) -> FastAPI:
    """``answerer_factory(cas)`` construit l'Answerer d'un cas d'usage (vrais
    modèles en production, doublures en test)."""
    app = FastAPI(title="Luciole", version=Versions().app)
    answerers: dict[str, Answerer] = {}
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="luciole-answer")
    tracer = tracer or JsonlTracer()

    def answer(request: QueryRequest) -> dict:  # toujours dans le fil de travail
        if request.use_case not in answerers:
            known = use_cases() if use_cases else None
            if known is not None and request.use_case not in known:
                raise HTTPException(
                    404, f"cas d'usage « {request.use_case} » inconnu ({', '.join(known)})"
                )
            answerers[request.use_case] = answerer_factory(request.use_case)
        conditions = [Condition(f.field, f.op, f.value) for f in request.filters]
        try:
            result = answerers[request.use_case].answer(
                request.question, conditions=conditions, document_ids=request.document_ids
            )
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return result.to_dict()

    @app.post("/api/rag/query", response_model=QueryResponse)
    async def rag_query(request: QueryRequest) -> dict:
        return await asyncio.get_running_loop().run_in_executor(worker, answer, request)

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "versions": Versions().as_dict(),
            "use_cases": use_cases() if use_cases else None,
            "loaded": {
                name: {
                    "embedding": getattr(a.retriever.embedder, "name", None),
                    "reranker": getattr(a.retriever.reranker, "name", None),
                    "llm": getattr(a.llm, "model", None),
                    "hardware": a.hardware.name,
                }
                for name, a in answerers.items()
            },
        }

    @app.get("/api/traces/{trace_id}")
    def trace(trace_id: str) -> dict:
        found = tracer.find(trace_id)
        if found is None:
            raise HTTPException(404, f"trace « {trace_id} » introuvable")
        return found

    return app


def _production_app() -> FastAPI:  # pragma: no cover - vrais modèles et services
    import argparse

    from luciole.core.profile import list_profiles
    from luciole.generation.ask import build

    tracer = JsonlTracer()
    stores = []

    def factory(use_case: str) -> Answerer:
        args = argparse.Namespace(
            profile=use_case,
            profiles_dir="configs/profiles",
            hardware=None,
            hardware_dir="configs/hardware",
            models="configs/models.yaml",
            store=None,
            no_opensearch=False,
        )
        answerer, store = build(args, tracer=tracer)
        stores.append(store)
        return answerer

    return create_app(factory, tracer=tracer, use_cases=lambda: list_profiles("configs/profiles"))


def __getattr__(name: str):  # « uvicorn luciole.api.app:app » construit l'app à la demande
    if name == "app":
        return _production_app()
    raise AttributeError(name)
