"""Réponse sourcée à une question : recherche, rédaction, gardes, verdict.

1. ``retrieve`` (recherche hybride du profil) ; aucun passage : la réponse
   ``no_results`` du profil, verdict ``INSUFFISANT``, sans appeler le LLM.
2. Les passages entrent dans la fenêtre du LLM (profil matériel), du mieux
   classé au moins bien classé, chacun sous son étiquette ``[Source: …]``.
3. Rédaction par le LLM, température et graine du profil.
4. Gardes déterministes (``guards.py``), dans l'ordre de V5 :
   - esquive : une passe élargie (``generation.deep``) ; gardée seulement si
     elle n'esquive plus ;
   - tronquée ou dégénérée : une relance sous ``retry_options`` ; gardée
     seulement si elle est propre ;
   - pages hors lot, nombres hors extraits : signalés en fin de réponse,
     jamais corrigés.
5. Verdict : ``INSUFFISANT`` (aucun passage, ou esquive), ``SUFFISANT``
   (toutes les gardes propres), ``PARTIEL`` sinon.

La confiance est le meilleur score du reranker parmi les passages fournis
(entre 0 et 1), nulle si le verdict est ``INSUFFISANT``. Chaque réponse a sa
trace complète (``trace.py``).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from luciole.core.hardware import HardwareProfile
from luciole.core.profile import RetrievalSpec, UseCaseProfile
from luciole.core.versions import Versions
from luciole.generation.guards import (
    GuardReport,
    SourcePages,
    cited_pages,
    is_degenerate,
    is_evasive,
    lot_pages,
    numbers_outside_extracts,
    pages_outside_lot,
)
from luciole.generation.llm import LLM, Completion
from luciole.generation.prompts import PromptSet
from luciole.retrieval.hybrid import HybridRetriever, Passage, RetrievalResult
from luciole.store.sqlite import Condition

Verdict = Literal["SUFFISANT", "PARTIEL", "INSUFFISANT"]


def format_passages(passages: Sequence[Passage], budget_tokens: int) -> tuple[str, list[Passage]]:
    """Passages sous leur étiquette, tant que la somme de leurs tokens tient
    dans ``budget_tokens`` (le premier passe toujours)."""
    blocks, used, total = [], [], 0
    for p in passages:
        cost = p.unit.token_count
        if used and total + cost > budget_tokens:
            break
        total += cost
        used.append(p)
        blocks.append(f"[Source: {p.citation}]\n{p.text.strip()}")
    return "\n\n---\n\n".join(blocks), used


def _sources(passages: Sequence[Passage]) -> list[SourcePages]:
    return [
        SourcePages(p.unit.file_name, p.unit.page_start, p.unit.page_end, p.unit.page_labels)
        for p in passages
    ]


def decide(report: GuardReport, has_passages: bool) -> Verdict:
    if not has_passages or report.evasive:
        return "INSUFFISANT"
    return "SUFFISANT" if report.clean else "PARTIEL"


@dataclass
class Answer:
    question: str
    answer: str
    verdict: Verdict
    confidence: float
    trace_id: str
    passages: list[Passage] = field(default_factory=list)
    guards: GuardReport = field(default_factory=GuardReport)
    trace: dict = field(default_factory=dict)

    def citations(self) -> list[dict]:
        """Passages fournis au LLM (sans leur texte) ; ``cited`` dit si la
        réponse cite une de leurs pages."""
        pages = cited_pages(self.answer)
        out = []
        for p in self.passages:
            d = p.to_dict()
            d.pop("text")
            own = lot_pages(_sources([p])).get(p.unit.file_name.lower(), set())
            d["cited"] = bool(own & pages)
            out.append(d)
        return out

    def to_dict(self) -> dict:
        """Contrat de ``POST /api/rag/query``."""
        return {
            "answer": self.answer,
            "citations": self.citations(),
            "verdict": self.verdict,
            "confidence": self.confidence,
            "trace_id": self.trace_id,
            "sources_used": sorted({p.unit.relative_path for p in self.passages}),
            "guards": self.guards.to_dict(),
        }


class Answerer:
    def __init__(
        self,
        profile: UseCaseProfile,
        prompts: PromptSet,
        retriever: HybridRetriever,
        llm: LLM,
        hardware: HardwareProfile,
        *,
        tracer=None,
    ) -> None:
        self.profile = profile
        self.prompts = prompts
        self.retriever = retriever
        self.llm = llm
        self.hardware = hardware
        self.tracer = tracer
        gen = profile.generation
        self.budget = hardware.llm_num_ctx - gen.max_tokens - gen.prompt_reserve_tokens
        if self.budget <= 0:
            raise ValueError(
                f"fenêtre du LLM trop petite ({hardware.llm_num_ctx} tokens) pour "
                f"{gen.max_tokens} tokens générés et {gen.prompt_reserve_tokens} de consignes"
            )

    # -- étapes ---------------------------------------------------------------
    def _deep_spec(self) -> RetrievalSpec:
        deep = self.profile.generation.deep.model_dump()
        return RetrievalSpec.model_validate({**self.profile.retrieval.model_dump(), **deep})

    def _options(self, retry: bool = False) -> dict:
        gen = self.profile.generation
        extra = gen.retry_options if retry else gen.options
        return {
            "temperature": gen.temperature,
            "seed": gen.seed,
            "num_predict": gen.max_tokens,
            **extra,
        }

    def _write(self, question: str, passages: Sequence[Passage], retry: bool = False):
        context, used = format_passages(passages, self.budget)
        prompt = self.prompts.render_answer(question, context)
        completion = self.llm.generate(self.prompts.system, prompt, options=self._options(retry))
        return completion, used

    def _evasive(self, text: str) -> bool:
        g = self.profile.generation.guards
        return is_evasive(text, g.evasion_max_position, g.evasion_min_length)

    def _degenerate(self, text: str) -> bool:
        g = self.profile.generation.guards
        return is_degenerate(
            text, g.degenerate_min_lines, g.degenerate_min_ratio, g.degenerate_max_repeats
        )

    # -- réponse --------------------------------------------------------------
    def answer(
        self,
        question: str,
        *,
        conditions: list[Condition] | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> Answer:
        started = time.monotonic()
        trace_id = uuid.uuid4().hex
        gen = self.profile.generation
        g = gen.guards
        steps: list[dict] = []

        def search(spec: RetrievalSpec | None) -> RetrievalResult:
            return self.retriever.retrieve(
                question, conditions=conditions, document_ids=document_ids, spec=spec
            )

        def note(kind: str, completion: Completion, used: Sequence[Passage]) -> None:
            steps.append(
                {
                    "step": kind,
                    "passages": len(used),
                    "prompt_tokens": completion.prompt_tokens,
                    "completion_tokens": completion.completion_tokens,
                    "truncated": completion.truncated,
                }
            )

        retrieval = search(None)
        searches = [retrieval.trace()]
        if not retrieval.passages:
            result = Answer(
                question,
                self.prompts.no_results.strip(),
                "INSUFFISANT",
                0.0,
                trace_id,
                guards=GuardReport(evasive=True),
            )
            return self._finish(result, searches, steps, started)

        completion, used = self._write(question, retrieval.passages)
        note("rédaction", completion, used)
        report = GuardReport()

        # 1. esquive -> passe élargie, gardée si elle n'esquive plus
        evasive = self._evasive(completion.text)
        if evasive and g.retry_on_evasion:
            deep = search(self._deep_spec())
            searches.append(deep.trace() | {"deep": True})
            if deep.passages:
                second, used2 = self._write(question, deep.passages)
                note("passe élargie", second, used2)
                if not self._evasive(second.text):
                    completion, used, evasive = second, used2, False
        report.evasive = evasive

        # 2. tronquée ou dégénérée -> une relance, gardée si elle est propre
        truncated, degenerate = completion.truncated, self._degenerate(completion.text)
        if (truncated or degenerate) and g.retry_on_loop:
            again, used3 = self._write(question, used, retry=True)
            note("relance", again, used3)
            if not (again.truncated or self._degenerate(again.text) or self._evasive(again.text)):
                completion, used, truncated, degenerate = again, used3, False, False
        report.truncated, report.degenerate = truncated, degenerate

        # 3. pages et nombres sans extrait -> signalés, jamais corrigés
        text = completion.text.strip()
        report.pages_outside = pages_outside_lot(text, lot_pages(_sources(used)), g.page_window)
        extracts = "\n".join(p.text for p in used)
        report.numbers_outside = numbers_outside_extracts(text, extracts, g.number_min_digits)
        if g.flag_in_answer:
            if report.pages_outside:
                text += (
                    "\n\n_Pages citées sans extrait correspondant dans le lot : "
                    + ", ".join(report.pages_outside)
                    + "._"
                )
            if report.numbers_outside:
                text += (
                    "\n\n_Nombres sans extrait correspondant dans le lot : "
                    + ", ".join(report.numbers_outside)
                    + "._"
                )

        verdict = decide(report, bool(used))
        scores = [p.rerank for p in used if p.rerank is not None]
        confidence = (
            0.0
            if verdict == "INSUFFISANT" or not scores
            else round(min(1.0, max(0.0, max(scores))), 3)
        )
        result = Answer(question, text, verdict, confidence, trace_id, list(used), report)
        return self._finish(result, searches, steps, started)

    def _finish(
        self, result: Answer, searches: list[dict], steps: list[dict], started: float
    ) -> Answer:
        result.trace = {
            "trace_id": result.trace_id,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "profile": self.profile.name,
            "hardware": self.hardware.name,
            "versions": Versions().as_dict(),
            "models": {
                "embedding": getattr(self.retriever.embedder, "name", None),
                "reranker": getattr(self.retriever.reranker, "name", None),
                "llm": getattr(self.llm, "model", None),
            },
            "question": result.question,
            "searches": searches,
            "generation": steps,
            "options": self._options(),
            "guards": result.guards.to_dict(),
            "verdict": result.verdict,
            "confidence": result.confidence,
            "answer": result.answer,
            "duration_s": round(time.monotonic() - started, 3),
        }
        if self.tracer is not None:
            self.tracer.write(result.trace)
        return result
