"""Outils des boucles, bâtis sur le RAG de l'étape 4.

- ``extract`` : le LLM rend un JSON conforme au schéma ``params.schema`` du
  profil (``prompts.yaml``, section ``extract``), validé ; un JSON invalide
  n'est ni deviné ni réparé : ``count`` vaut 0 et les règles choisissent la
  suite (par exemple la recherche directe).
- ``compose`` : questions de recherche tirées d'une extraction par les
  gabarits du profil (section ``questions``), sans LLM.
- ``retrieve`` : recherche hybride du profil, filtrée comme la question ;
  ``mode: deep`` prend les réglages de la passe élargie (``generation.deep``) ;
  ``per: question`` cherche la demande puis chaque question composée et
  entrelace les résultats (rang 1 de chaque liste, puis rang 2…), sans
  doublon : les premiers passages de la demande restent donc en tête ; la
  limite des documents secondaires vaut pour l'ensemble.
- ``synthesize`` : rédaction par le LLM sur les passages du dernier
  ``retrieve``, avec les gardes de l'étape 4 (relance d'une réponse tronquée
  ou qui boucle, pages et nombres hors extraits signalés) ; ``framing``
  ajoute les cadrages du profil remplis par les notes de la boucle
  (extractions, fiche du dossier, comparables).
- ``verdict`` : décision déterministe ; garde la meilleure réponse rédigée
  jusqu'ici (verdict, puis confiance). Une passe élargie ne peut donc jamais
  remplacer une meilleure réponse par une moins bonne.

- ``filter``, ``aggregate``, ``compare`` : sur le magasin structuré, sans LLM
  (``records.py``).

``extract`` avec ``per: document`` fait une extraction par document retenu par
``filter``. ``notify`` arrivera avec la boucle qui en aura besoin.
"""

from __future__ import annotations

import string
from collections.abc import Mapping, Sequence

from luciole.agents.engine import LoopAgent, State
from luciole.agents.records import (
    AGGREGATE_FIELDS,
    AggregateTool,
    CompareTool,
    FilterTool,
    read_document,
)
from luciole.core.profile import Loop
from luciole.generation.answer import Answerer
from luciole.generation.llm import extract
from luciole.generation.prompts import PromptSet, as_text, template_fields
from luciole.retrieval.hybrid import Passage
from luciole.retrieval.lot import is_secondary


def interleave(
    lists: Sequence[Sequence[Passage]],
    secondary: Sequence[str] = (),
    cap: int = 0,
) -> tuple[list[Passage], int]:
    """Rang 1 de chaque liste, puis rang 2…, sans répéter un même passage rendu.

    La règle du lot vaut pour l'ensemble fusionné : au plus ``cap`` passages
    des documents ``secondary`` (résumé non technique…), sinon chaque question
    en apporterait ``cap``. Rend aussi le nombre de passages écartés ainsi."""
    seen: set[str] = set()
    merged: list[Passage] = []
    kept_secondary = capped = 0
    for rank in range(max((len(x) for x in lists), default=0)):
        for passages in lists:
            if rank >= len(passages):
                continue
            p = passages[rank]
            if p.unit.chunk_id in seen:
                continue
            seen.add(p.unit.chunk_id)
            if cap and secondary and is_secondary(p.unit.file_name, secondary):
                if kept_secondary >= cap:
                    capped += 1
                    continue
                kept_secondary += 1
            merged.append(p)
    return merged, capped


class ExtractTool:
    """``per: request`` (défaut) : une extraction sur la demande. ``per:
    document`` : une par document retenu par ``filter``, dans l'ordre, sur son
    texte entier ; chaque résultat porte ``source`` (chemin) et ``rang``. Un
    document trop long pour la fenêtre du LLM ou une sortie non conforme est
    signalé et laissé de côté, jamais tronqué ni deviné."""

    def __init__(self, answerer: Answerer) -> None:
        self.answerer = answerer

    def _one(self, spec, prompt: str) -> dict:
        result = extract(
            self.answerer.llm,
            spec.system,
            prompt,
            spec.model(),
            options=self.answerer.llm_options(),
        )
        return result.model_dump()

    def run(self, params: Mapping[str, object], state: State) -> dict:
        name = str(params.get("schema", ""))
        spec = self.answerer.prompts.extract.get(name)
        if spec is None:
            raise ValueError(f"extract : schéma « {name} » absent de prompts.yaml")
        per = params.get("per", "request")
        if per == "request":
            try:
                state.extracted[name] = self._one(spec, spec.prompt.format(question=state.question))
            except ValueError as e:
                state.count = 0
                return {"schema": name, "error": str(e)[:300]}
            state.count = 1
            return {"schema": name, "values": state.extracted[name]}
        if per != "document":
            raise ValueError(f"extract : per « {per} » inconnu (request ou document)")
        store = self.answerer.retriever.store
        values, errors = [], []
        for rank, doc in enumerate(state.documents, start=1):
            chunks = read_document(store, doc.document_id)
            cost = sum(c.token_count for c in chunks)
            if cost > self.answerer.budget:
                errors.append({"source": doc.relative_path, "error": f"{cost} tokens, trop long"})
                continue
            text = "\n\n".join(c.text.strip() for c in chunks)
            prompt = spec.prompt.format_map(
                {"question": state.question, "document": text, "source": doc.relative_path}
            )
            try:
                values.append({"source": doc.relative_path, "rang": rank} | self._one(spec, prompt))
            except ValueError as e:
                errors.append({"source": doc.relative_path, "error": str(e)[:300]})
        state.extracted[name] = values
        state.count = len(values)
        return {"schema": name, "documents": len(state.documents), "values": values} | (
            {"errors": errors} if errors else {}
        )


class ComposeTool:
    """Une question pour l'objet (gabarit ``first``, ``objet_{type_objet}``
    par exemple, sinon ``fallback``), puis une par élément des listes nommées
    dans ``each`` : ``{aspects: aspect}`` passe chaque aspect au gabarit
    ``aspect`` sous le champ ``{aspect}``. Au plus ``max_questions`` : les
    dernières listes de ``each`` cèdent leur place en premier."""

    def __init__(self, answerer: Answerer) -> None:
        self.answerer = answerer

    def run(self, params: Mapping[str, object], state: State) -> dict:
        source = str(params.get("from", ""))
        values = state.extracted.get(source)
        if values is None:
            raise ValueError(f"compose : aucune extraction « {source} » avant ce nœud")
        templates = self.answerer.prompts.questions
        plain = {k: as_text(v) for k, v in values.items()}
        first = string.Formatter().vformat(str(params.get("first", "")), (), plain)
        key = first if first in templates else str(params.get("fallback", ""))
        if key not in templates:
            raise ValueError(f"compose : gabarit « {first} » ni repli dans prompts.yaml")
        questions = [templates[key].format(**plain).strip()]
        each = dict(params.get("each", {}) or {})
        groups: list[list[str]] = []
        for field_name, template in each.items():
            if template not in templates:
                raise ValueError(f"compose : gabarit « {template} » absent de prompts.yaml")
            slot = template  # « aspect » reçoit chaque aspect, « motif » chaque motif
            groups.append(
                [
                    templates[template].format(**plain, **{slot: item}).strip()
                    for item in values.get(field_name, [])
                    if str(item).strip()
                ]
            )
        limit = int(params.get("max_questions", 6))
        while len(questions) + sum(map(len, groups)) > limit and any(groups):
            # Une liste plus haute dans « each » garde sa place plus longtemps.
            last = max(i for i, g in enumerate(groups) if g)
            groups[last].pop()
        for group in groups:
            questions.extend(group)
        state.questions = questions[:limit]
        state.count = len(state.questions)
        return {"questions": state.questions}


class RetrieveTool:
    def __init__(self, answerer: Answerer) -> None:
        self.answerer = answerer

    def run(self, params: Mapping[str, object], state: State) -> dict:
        mode = params.get("mode", "normal")
        if mode not in ("normal", "deep"):
            raise ValueError(f"retrieve : mode « {mode} » inconnu (normal ou deep)")
        spec = self.answerer.deep_spec() if mode == "deep" else None
        per = params.get("per", "request")
        if per not in ("request", "question"):
            raise ValueError(f"retrieve : per « {per} » inconnu (request ou question)")
        queries = [state.question]
        if per == "question":
            queries += [q for q in state.questions if q != state.question]
        top_k = params.get("top_k")
        found = []
        for query in queries:
            result = self.answerer.retriever.retrieve(
                query,
                conditions=state.conditions,
                document_ids=state.document_ids,
                spec=spec,
                top_k=int(top_k) if top_k else None,
            )
            state.searches.append(result.trace() | {"deep": mode == "deep"})
            found.append(result.passages)
        lot = spec or self.answerer.retriever.spec
        state.passages, capped = interleave(
            found, lot.secondary_documents, lot.secondary_cap if len(found) > 1 else 0
        )
        state.count = len(state.passages)
        return {
            "mode": mode,
            "queries": len(queries),
            "passages": state.count,
            "capped_secondary": capped,
        }


def framing_names(params: Mapping[str, object]) -> list[str]:
    value = params.get("framing") or []
    return [value] if isinstance(value, str) else [str(v) for v in value]


class SynthesizeTool:
    def __init__(self, answerer: Answerer) -> None:
        self.answerer = answerer

    def run(self, params: Mapping[str, object], state: State) -> dict:
        prompts = self.answerer.prompts
        parts = [
            prompts.render_framing(name, state.extracted[name])
            for name in framing_names(params)
            if name in state.extracted
        ]
        framing = "\n\n".join(p for p in parts if p)
        draft, calls = self.answerer.draft(state.question, state.passages, framing=framing)
        state.latest = draft
        state.generation.extend(calls)
        return {"verdict": draft.verdict, "llm_calls": len(calls)}


class VerdictTool:
    def run(self, params: Mapping[str, object], state: State) -> dict:
        if state.latest is None:
            return {"kept": False, "reason": "aucune réponse rédigée"}
        kept = state.keep_best(state.latest)
        return {"latest": state.latest.verdict, "kept": kept}


def _notes(loop: Loop, prompts: PromptSet, dossier_fields: set[str]) -> dict[str, set[str]]:
    """Notes que les nœuds de la boucle écrivent, avec leurs champs."""
    notes: dict[str, set[str]] = {}
    for node in loop.nodes.values():
        params = node.params
        if node.type == "extract" and params.get("schema") in prompts.extract:
            notes[str(params["schema"])] = set(prompts.extract[str(params["schema"])].fields) | (
                {"source", "rang"} if params.get("per") == "document" else set()
            )
        elif node.type == "filter":
            notes[str(params.get("as", "dossier"))] = dossier_fields | {"dossier_id"}
        elif node.type == "aggregate":
            notes[str(params.get("as", "comparables"))] = set(AGGREGATE_FIELDS)
    return notes


def check_references(loop: Loop, prompts: PromptSet, dossier_fields=frozenset()) -> None:
    """Refuse au chargement une boucle qui nomme un schéma, un gabarit ou un
    cadrage absent de ``prompts.yaml``, ou un cadrage qui lit un champ que sa
    note n'a pas, plutôt qu'à la première question."""
    problems = []
    notes = _notes(loop, prompts, set(dossier_fields))
    for name, node in loop.nodes.items():
        params = node.params
        if node.type == "extract" and params.get("schema") not in prompts.extract:
            problems.append(f"{name} : schéma « {params.get('schema')} » absent")
        if node.type == "compose":
            if params.get("from") not in prompts.extract:
                problems.append(f"{name} : extraction « {params.get('from')} » absente")
            if params.get("fallback") and params["fallback"] not in prompts.questions:
                problems.append(f"{name} : gabarit « {params['fallback']} » absent")
            for template in dict(params.get("each", {}) or {}).values():
                if template not in prompts.questions:
                    problems.append(f"{name} : gabarit « {template} » absent")
        if node.type == "compare" and params.get("from") not in prompts.extract:
            problems.append(f"{name} : extraction « {params.get('from')} » absente")
        if node.type == "synthesize" and framing_names(params):
            if "{framing}" not in prompts.answer:
                problems.append(f"{name} : le gabarit answer n'emploie pas {{framing}}")
            for note in framing_names(params):
                if note not in notes:
                    problems.append(f"{name} : cadrage « {note} » sans nœud qui l'écrit")
                elif note not in prompts.framing:
                    problems.append(f"{name} : cadrage « {note} » absent de prompts.yaml")
                else:
                    unknown = template_fields(prompts.framing[note]) - notes[note]
                    if unknown:
                        problems.append(
                            f"{name} : cadrage « {note} » : champ(s) inconnu(s) {sorted(unknown)}"
                        )
    if problems:
        raise ValueError(f"boucle « {loop.name} » / prompts.yaml : " + " ; ".join(problems))


def build_agent(loop: Loop, answerer: Answerer, *, tracer=None) -> LoopAgent:
    """La boucle d'un cas d'usage avec les outils du RAG de ce cas."""
    tools = {
        "filter": FilterTool(answerer),
        "aggregate": AggregateTool(answerer),
        "compare": CompareTool(answerer),
        "extract": ExtractTool(answerer),
        "compose": ComposeTool(answerer),
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
    agent = LoopAgent(loop, tools, answerer.empty, meta=meta, tracer=tracer)
    check_references(loop, answerer.prompts, set(answerer.profile.entities.dossier))
    return agent
