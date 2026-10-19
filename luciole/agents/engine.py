"""Moteur de boucles : la machine à états de ``loop.yaml``, tenue par le code.

À chaque nœud, le moteur :

1. arrête la boucle si le nœud est terminal (``stopped_reason`` = son nom) ;
2. arrête la boucle sur ``on_budget`` si le plafond ``max_steps`` est atteint,
   ou si un nœud ``refine: true`` est demandé sans raffinement restant ;
3. exécute l'outil du nœud (``tools[type]``), qui met l'état à jour ;
4. évalue les règles du nœud dans l'ordre (``rules.py``) : la première qui
   matche choisit la suite, sinon ``default``.

Le LLM n'intervient que dans les outils, pour des tâches bornées (rédiger,
extraire un JSON validé) ; il ne choisit jamais la suite ni l'arrêt
(``agent-harness-architecture.md``, post-mortem Belacom). Chaque exécution
écrit une trace complète : nœuds traversés, état après chaque outil, règle qui
a matché, recherches et appels au LLM, durées.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from luciole.agents.rules import STATE_FIELDS, parse
from luciole.core.profile import TERMINAL_NODE_TYPE, Loop, Node
from luciole.core.versions import Versions
from luciole.generation.answer import Answer
from luciole.retrieval.hybrid import Passage
from luciole.store.sqlite import Condition

BUDGET_EXHAUSTED = "budget_epuise"
_VERDICT_RANK = {"INSUFFISANT": 0, "PARTIEL": 1, "SUFFISANT": 2}


@dataclass
class State:
    """Ce que les outils lisent et écrivent ; les règles n'en voient que
    ``fields()`` (liste blanche ``STATE_FIELDS``)."""

    question: str
    conditions: list[Condition] | None = None
    document_ids: Sequence[str] | None = None
    passages: list[Passage] = field(default_factory=list)
    #: Extractions typées, par nom de schéma (outil ``extract``).
    extracted: dict[str, dict] = field(default_factory=dict)
    #: Questions de recherche composées (outil ``compose``).
    questions: list[str] = field(default_factory=list)
    latest: Answer | None = None
    best: Answer | None = None
    count: int | None = None
    refinements_left: int = 0
    step: int = 0
    steps_left: int = 0
    searches: list[dict] = field(default_factory=list)
    generation: list[dict] = field(default_factory=list)

    def keep_best(self, candidate: Answer) -> bool:
        """Garde la meilleure réponse : verdict d'abord, confiance ensuite ; à
        égalité, la première reste. Rend vrai si ``candidate`` est gardée."""
        if self.best is None or _rank(candidate) > _rank(self.best):
            self.best = candidate
            return True
        return False

    def fields(self) -> dict[str, object]:
        best = self.best
        values = {
            "verdict": best.verdict if best else None,
            "confidence": best.confidence if best else None,
            "evasive": best.guards.evasive if best else None,
            "count": self.count,
            "refinements_left": self.refinements_left,
            "steps_left": self.steps_left,
            "step": self.step,
        }
        assert set(values) == set(STATE_FIELDS)
        return values


def _rank(answer: Answer) -> tuple[int, float]:
    return (_VERDICT_RANK[answer.verdict], answer.confidence)


class Tool(Protocol):
    def run(self, params: Mapping[str, object], state: State) -> dict:
        """Exécute l'outil et met ``state`` à jour ; rend un résumé pour la trace."""
        ...


@dataclass
class AgentResult:
    answer: Answer
    stopped_reason: str
    trace_id: str
    path: list[str]
    matched_rules: list[dict]
    steps_used: int
    trace: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Contrat de ``POST /api/agent/query`` : celui de ``/api/rag/query``
        plus le chemin suivi dans la boucle."""
        return self.answer.to_dict() | {
            "trace_id": self.trace_id,
            "stopped_reason": self.stopped_reason,
            "path": self.path,
            "steps_used": self.steps_used,
            "matched_rules": self.matched_rules,
        }


class LoopAgent:
    """Exécute une boucle validée avec les outils fournis.

    ``empty(question, trace_id)`` donne la réponse quand aucun outil n'en a
    rédigé (dossier vide, arrêt avant rédaction) ; ``meta`` complète la trace
    (profil, matériel, modèles)."""

    def __init__(
        self,
        loop: Loop,
        tools: Mapping[str, Tool],
        empty: Callable[[str, str], Answer],
        *,
        meta: Mapping[str, object] | None = None,
        tracer=None,
    ) -> None:
        missing = sorted(
            {n.type for n in loop.nodes.values() if n.type != TERMINAL_NODE_TYPE} - set(tools)
        )
        if missing:
            raise ValueError(
                f"boucle « {loop.name} » : outil(s) {', '.join(missing)} pas encore "
                "exécutable(s) par le moteur"
            )
        self.loop = loop
        self.tools = dict(tools)
        self.empty = empty
        self.meta = dict(meta or {})
        self.tracer = tracer
        self._rules = {
            name: [(parse(r.when), r) for r in node.rules] for name, node in loop.nodes.items()
        }

    def answer(self, question: str, **kwargs) -> Answer:
        """Même usage que ``Answerer.answer`` (banc d'essai) : la réponse finale."""
        return self.run(question, **kwargs).answer

    def run(
        self,
        question: str,
        *,
        conditions: list[Condition] | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> AgentResult:
        started = time.monotonic()
        trace_id = uuid.uuid4().hex
        loop = self.loop
        state = State(
            question,
            conditions=conditions,
            document_ids=document_ids,
            refinements_left=loop.max_refinements,
        )
        path: list[str] = []
        steps: list[dict] = []
        matched: list[dict] = []
        current = loop.start
        stopped: str | None = None
        while stopped is None:
            node = loop.nodes[current]
            if node.type == TERMINAL_NODE_TYPE:
                path.append(current)
                stopped = current
                break
            refine = bool(node.params.get("refine"))
            if state.step >= loop.max_steps or (refine and state.refinements_left <= 0):
                why = "max_steps" if state.step >= loop.max_steps else "max_refinements"
                steps.append({"node": current, "stopped": why})
                stopped = loop.on_budget or BUDGET_EXHAUSTED
                if loop.on_budget:
                    path.append(loop.on_budget)
                break
            state.step += 1
            state.steps_left = loop.max_steps - state.step
            if refine:
                state.refinements_left -= 1
            path.append(current)
            t0 = time.monotonic()
            params = {k: v for k, v in node.params.items() if k != "refine"}
            detail = self.tools[node.type].run(params, state)
            current, rule = self._next(current, node, state)
            if rule is not None:
                matched.append(rule)
            steps.append(
                {
                    "node": path[-1],
                    "type": node.type,
                    "detail": detail,
                    "state": state.fields(),
                    "rule": rule,
                    "next": current,
                    "duration_s": round(time.monotonic() - t0, 3),
                }
            )

        # Arrêt avant tout verdict (plafond) : la dernière rédaction, sinon la réponse vide.
        answer = state.best or state.latest or self.empty(question, trace_id)
        answer.trace_id = trace_id
        result = AgentResult(
            answer,
            stopped,
            trace_id,
            path,
            matched,
            steps_used=state.step,
        )
        result.trace = self._trace(result, state, steps, started)
        answer.trace = result.trace
        if self.tracer is not None:
            self.tracer.write(result.trace)
        return result

    def _next(self, name: str, node: Node, state: State) -> tuple[str, dict | None]:
        values = state.fields()
        for condition, rule in self._rules[name]:
            if condition(values):
                return rule.then, {
                    "node": name,
                    "when": rule.when,
                    "then": rule.then,
                    "reason": rule.reason,
                }
        assert node.default is not None, "aucune règle ni suite : refusé par Loop"
        return node.default, None

    def _trace(self, result: AgentResult, state: State, steps: list[dict], started) -> dict:
        answer = result.answer
        return {
            "trace_id": result.trace_id,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "kind": "agent",
            "loop": self.loop.name,
            **self.meta,
            "versions": Versions().as_dict(),
            "question": state.question,
            "path": result.path,
            "steps": steps,
            "matched_rules": result.matched_rules,
            "searches": state.searches,
            "generation": state.generation,
            "guards": answer.guards.to_dict(),
            "verdict": answer.verdict,
            "confidence": answer.confidence,
            "stopped_reason": result.stopped_reason,
            "answer": answer.answer,
            "duration_s": round(time.monotonic() - started, 3),
        }
