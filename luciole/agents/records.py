"""Outils des boucles sur le magasin structuré (dossiers et documents).

- ``filter`` : retrouve le dossier de la demande et ses documents dans l'ordre
  du profil (``document_order``). Le dossier vient des conditions de l'appel
  (``dossier_id`` donné par l'API ou le banc), sinon du champ ``match`` du
  dossier (le client, par exemple) écrit en toutes lettres dans la demande,
  sans accent ni casse. Ni zéro ni deux dossiers ne sont devinés : ``count``
  vaut alors 0. Avec ``read: true``, les passages deviennent les documents
  entiers (parents), dans l'ordre : la chronologie d'une suite de réunions.
  La fiche du dossier devient une note (``as``, ``dossier`` par défaut).
- ``aggregate`` : compte les dossiers qui partagent les champs ``same`` du
  dossier retenu (le produit), par valeur de ``field`` (l'issue) restreinte à
  ``values`` ; ces dossiers deviennent les comparables.
- ``compare`` : cherche dans les comparables chaque élément des listes
  ``fields`` d'une extraction (les objections), au plus ``max_queries``
  recherches de ``top_k`` passages, ajoutés après les passages du dossier.

Aucun de ces outils n'appelle le LLM.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

from luciole.agents.engine import State
from luciole.core.chunk import Chunk
from luciole.retrieval.hybrid import Passage
from luciole.store.sqlite import Condition

#: Champs de la note écrite par ``aggregate``.
AGGREGATE_FIELDS = ("filtre", "comptes", "dossiers")


def normalize(text: str) -> str:
    """Minuscules, sans accent ni ponctuation, espaces simples ; une mention
    entre parenthèses (« (fictif) ») est retirée."""
    text = re.sub(r"\([^)]*\)", " ", str(text))
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def read_document(store, document_id: str) -> list[Chunk]:
    """Le document entier : ses parents dans l'ordre, sinon ses enfants."""
    return store.chunks_of(document_id, "parent") or store.chunks_of(document_id, "child")


def whole(chunk: Chunk) -> Passage:
    """Un morceau lu en entier (pas trouvé par la recherche) comme passage."""
    return Passage(hit=chunk, unit=chunk, rrf=0.0, ranks={})


class FilterTool:
    def __init__(self, answerer) -> None:
        self.answerer = answerer

    def _dossiers(self, params: Mapping[str, object], state: State) -> list[str]:
        store = self.answerer.retriever.store
        if state.conditions:
            docs = store.select("document", state.conditions)
            return sorted({d.dossier_id for d in docs})
        field = str(params.get("match", ""))
        if not field:
            return []
        question = f" {normalize(state.question)} "
        found = []
        for dossier in store.select("dossier"):
            name = normalize(dossier.metadata.get(field) or "")
            if name and f" {name} " in question:
                found.append(dossier.dossier_id)
        return found

    def run(self, params: Mapping[str, object], state: State) -> dict:
        store = self.answerer.retriever.store
        found = self._dossiers(params, state)
        if len(found) != 1:
            state.count = 0
            state.documents = []
            return {"dossiers": found, "reason": "aucun dossier" if not found else "plusieurs"}
        [dossier_id] = found
        dossier = store.get_dossier(dossier_id)
        state.dossier_id = dossier_id
        state.documents = store.select("document", [Condition("dossier_id", "eq", dossier_id)])
        state.conditions = [Condition("dossier_id", "eq", dossier_id)]
        state.extracted[str(params.get("as", "dossier"))] = {
            "dossier_id": dossier_id,
            **dossier.metadata,
        }
        if params.get("read"):
            state.passages = [
                whole(c) for d in state.documents for c in read_document(store, d.document_id)
            ]
        state.count = len(state.documents)
        return {
            "dossier": dossier_id,
            "documents": [d.relative_path for d in state.documents],
            "passages": len(state.passages),
        }


class AggregateTool:
    def __init__(self, answerer) -> None:
        self.answerer = answerer

    def run(self, params: Mapping[str, object], state: State) -> dict:
        store = self.answerer.retriever.store
        target = state.dossier_id
        dossier = store.get_dossier(target) if target else None
        if dossier is None:
            raise ValueError("aggregate : aucun dossier retenu avant ce nœud (outil filter)")
        same = [str(f) for f in params.get("same", [])]
        field = str(params.get("field", ""))
        values = list(params.get("values", []))
        conditions = [Condition(f, "eq", dossier.metadata.get(f)) for f in same]
        if values:
            conditions.append(Condition(field, "in", values))
        others = [d for d in store.select("dossier", conditions) if d.dossier_id != target]
        counts: dict[str, int] = {}
        for d in others:
            key = str(d.metadata.get(field))
            counts[key] = counts.get(key, 0) + 1
        state.comparables = [d.dossier_id for d in others]
        state.extracted[str(params.get("as", "comparables"))] = {
            "filtre": " ; ".join(f"{f} = {dossier.metadata.get(f)}" for f in same) or "aucun",
            "comptes": [f"{k} : {n}" for k, n in sorted(counts.items())],
            "dossiers": [f"{d.dossier_id} ({d.metadata.get(field)})" for d in others],
        }
        state.count = len(others)
        return {"comparables": state.comparables, "counts": counts}


class CompareTool:
    def __init__(self, answerer) -> None:
        self.answerer = answerer

    def run(self, params: Mapping[str, object], state: State) -> dict:
        source = str(params.get("from", ""))
        notes = state.extracted.get(source)
        if notes is None:
            state.count = 0
            return {"queries": 0, "reason": f"aucune extraction « {source} »"}
        notes = notes if isinstance(notes, list) else [notes]
        queries: list[str] = []
        for note in notes:
            for f in params.get("fields", []):
                for item in note.get(str(f), []) or []:
                    item = str(item).strip()
                    if item and item not in queries:
                        queries.append(item)
        queries = queries[: int(params.get("max_queries", 4))]
        if not state.comparables or not queries:
            state.count = 0
            return {"queries": 0, "reason": "aucun comparable ou rien à comparer"}
        where = [Condition("dossier_id", "in", state.comparables)]
        seen = {p.unit.chunk_id for p in state.passages}
        added: list[Passage] = []
        for query in queries:
            result = self.answerer.retriever.retrieve(
                query, conditions=where, top_k=int(params.get("top_k", 2))
            )
            state.searches.append(result.trace() | {"compare": True})
            for p in result.passages:
                if p.unit.chunk_id not in seen:
                    seen.add(p.unit.chunk_id)
                    added.append(p)
        state.passages = [*state.passages, *added]
        state.count = len(added)
        return {
            "queries": queries,
            "passages": [p.citation for p in added],
        }
