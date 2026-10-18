"""Fusion de listes classées par Reciprocal Rank Fusion (RRF).

Chaque liste donne à son élément de rang ``r`` (à partir de 1) le score
``1 / (k + r)`` ; les scores s'additionnent d'une liste à l'autre. Seuls les
rangs comptent : les scores bruts de BM25 et du cosinus ne se comparent pas.
À score égal, l'ordre est celui de la première apparition (stable).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass
class Fused:
    chunk_id: str
    score: float
    #: Rang (à partir de 1) dans chaque liste où l'élément apparaît.
    ranks: dict[str, int] = field(default_factory=dict)


def rrf(lists: Mapping[str, Sequence[str]], k: int) -> list[Fused]:
    """Fusionne des listes nommées d'identifiants (meilleur d'abord)."""
    if k < 1:
        raise ValueError("rrf : k doit être au moins 1")
    fused: dict[str, Fused] = {}
    for name, ids in lists.items():
        seen: set[str] = set()
        for rank, chunk_id in enumerate(ids, start=1):
            if chunk_id in seen:  # un doublon dans une même liste ne compte qu'une fois
                continue
            seen.add(chunk_id)
            item = fused.setdefault(chunk_id, Fused(chunk_id, 0.0))
            item.score += 1.0 / (k + rank)
            item.ranks[name] = rank
    return sorted(fused.values(), key=lambda f: f.score, reverse=True)
