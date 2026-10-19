"""Outils des boucles, bâtis sur le RAG de l'étape 4.

- ``retrieve`` : recherche hybride du profil, filtrée comme la question ;
  ``mode: deep`` prend les réglages de la passe élargie (``generation.deep``).
- ``synthesize`` : rédaction par le LLM sur les passages du dernier
  ``retrieve``, avec les gardes de l'étape 4 (relance d'une réponse tronquée
  ou qui boucle, pages et nombres hors extraits signalés).
- ``verdict`` : décision déterministe ; garde la meilleure réponse rédigée
  jusqu'ici (verdict, puis confiance). Une passe élargie ne peut donc jamais
  remplacer une meilleure réponse par une moins bonne.

Les outils ``filter``, ``aggregate``, ``extract``, ``compare`` et ``notify``
arrivent avec les boucles qui en ont besoin.
"""

from __future__ import annotations

from collections.abc import Mapping

from luciole.agents.engine import LoopAgent, State
from luciole.core.profile import Loop
from luciole.generation.answer import Answerer


class RetrieveTool:
    def __init__(self, answerer: Answerer) -> None:
        self.answerer = answerer

    def run(self, params: Mapping[str, object], state: State) -> dict:
        mode = params.get("mode", "normal")
        if mode not in ("normal", "deep"):
            raise ValueError(f"retrieve : mode « {mode} » inconnu (normal ou deep)")
        spec = self.answerer.deep_spec() if mode == "deep" else None
        result = self.answerer.retriever.retrieve(
            state.question,
            conditions=state.conditions,
            document_ids=state.document_ids,
            spec=spec,
        )
        state.passages = list(result.passages)
        state.count = len(state.passages)
        state.searches.append(result.trace() | {"deep": mode == "deep"})
        return {"mode": mode, "passages": state.count}


class SynthesizeTool:
    def __init__(self, answerer: Answerer) -> None:
        self.answerer = answerer

    def run(self, params: Mapping[str, object], state: State) -> dict:
        draft, calls = self.answerer.draft(state.question, state.passages)
        state.latest = draft
        state.generation.extend(calls)
        return {"verdict": draft.verdict, "llm_calls": len(calls)}


class VerdictTool:
    def run(self, params: Mapping[str, object], state: State) -> dict:
        if state.latest is None:
            return {"kept": False, "reason": "aucune réponse rédigée"}
        kept = state.keep_best(state.latest)
        return {"latest": state.latest.verdict, "kept": kept}


def build_agent(loop: Loop, answerer: Answerer, *, tracer=None) -> LoopAgent:
    """La boucle d'un cas d'usage avec les outils du RAG de ce cas."""
    tools = {
        "retrieve": RetrieveTool(answerer),
        "synthesize": SynthesizeTool(answerer),
        "verdict": VerdictTool(),
    }
    meta = {
        "profile": answerer.profile.name,
        "hardware": answerer.hardware.name,
        "models": {
            "embedding": getattr(answerer.retriever.embedder, "name", None),
            "reranker": getattr(answerer.retriever.reranker, "name", None),
            "llm": getattr(answerer.llm, "model", None),
        },
        "options": answerer.llm_options(),
    }
    return LoopAgent(loop, tools, answerer.empty, meta=meta, tracer=tracer)
