"""Doublures partagées par les tests unitaires : rien n'est téléchargé, aucun
service n'est appelé."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from luciole.index.embedder import normalize

_WORD_RE = re.compile(r"\w+")


class FakeEmbedder:
    """Sac de mots haché dans ``dimension`` cases : deux textes qui partagent
    des mots ont des vecteurs proches, ce qui suffit à vérifier une recherche."""

    def __init__(self, dimension: int = 64, name: str = "faux-embedder") -> None:
        self.dimension = dimension
        self.name = name
        self.calls = 0
        self.texts = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        self.texts += len(texts)
        out = []
        for text in texts:
            vector = [0.0] * self.dimension
            for word in _WORD_RE.findall(text.lower()):
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                vector[h % self.dimension] += 1.0
            out.append(normalize(vector))
        return out


class _Indices:
    def __init__(self, store: FakeOpenSearch) -> None:
        self._store = store

    def exists(self, index: str) -> bool:
        return index in self._store.indexes

    def create(self, index: str, body: dict) -> None:
        self._store.indexes[index] = {}
        self._store.mappings[index] = body


class FakeOpenSearch:
    """Ce que ``OpenSearchIndex`` appelle, en mémoire. La recherche compte les
    mots communs (pas de BM25) : elle sert à vérifier le câblage, pas le
    classement."""

    def __init__(self) -> None:
        self.indexes: dict[str, dict[str, dict]] = {}
        self.mappings: dict[str, dict] = {}
        self.indices = _Indices(self)

    @staticmethod
    def _matches(doc: dict, query: dict | None) -> bool:
        if not query:
            return True
        field, value = next(iter(query["term"].items()))
        return doc.get(field) == value

    def delete_by_query(self, index: str, body: dict, **_: object) -> dict:
        docs = self.indexes[index]
        gone = [i for i, d in docs.items() if self._matches(d, body["query"])]
        for i in gone:
            del docs[i]
        return {"deleted": len(gone)}

    def bulk(self, body: list[dict], **_: object) -> dict:
        items = []
        for action, source in zip(body[::2], body[1::2], strict=True):
            meta = action["index"]
            props = self.mappings[meta["_index"]]["mappings"]["properties"]
            unknown = set(source) - set(props)
            if unknown:  # « dynamic: strict » refuse les champs non déclarés
                items.append({"index": {"error": f"champs inconnus {sorted(unknown)}"}})
                continue
            self.indexes[meta["_index"]][meta["_id"]] = source
            items.append({"index": {"result": "created"}})
        return {"errors": any("error" in i["index"] for i in items), "items": items}

    def count(self, index: str, body: dict | None = None) -> dict:
        query = body["query"] if body else None
        return {"count": sum(1 for d in self.indexes[index].values() if self._matches(d, query))}

    def search(self, index: str, body: dict) -> dict:
        bool_query = body["query"]["bool"]
        words = set(_WORD_RE.findall(bool_query["must"][0]["multi_match"]["query"].lower()))
        allowed = None
        for f in bool_query["filter"]:
            allowed = set(f["terms"]["document_id"])
        hits = []
        for doc in self.indexes[index].values():
            if allowed is not None and doc["document_id"] not in allowed:
                continue
            score = len(words & set(_WORD_RE.findall(doc["text"].lower())))
            if score:
                hits.append({"_score": float(score), "_source": doc})
        hits.sort(key=lambda h: -h["_score"])
        return {"hits": {"hits": hits[: body["size"]]}}
