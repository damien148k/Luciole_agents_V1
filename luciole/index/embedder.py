"""Contrat d'un embedder : des textes en entrée, des vecteurs normalisés en
sortie. Le vrai modèle (BGE-M3, étape 3b) et le faux des tests le respectent ;
le pipeline et les index ne connaissent que ce contrat (rien de lourd n'est
importé pour les tests unitaires).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from luciole.core.versions import EMBEDDING_PROFILE_VERSION, INDEX_SCHEMA_VERSION


@runtime_checkable
class Embedder(Protocol):
    #: Nom du modèle, révision comprise (« BAAI/bge-m3@5617a9f »).
    name: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Un vecteur de ``dimension`` flottants, de norme 1, par texte."""
        ...


def normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else [0.0 for _ in vector]


def index_key(embedder: Embedder) -> str:
    """Ce qui rend un index incompatible avec un autre : schéma, profil
    d'embedding, modèle, dimension. Changer l'un des quatre impose de
    réindexer ; le pipeline compare cette clé à celle notée dans le magasin."""
    return (
        f"index{INDEX_SCHEMA_VERSION}/embedding{EMBEDDING_PROFILE_VERSION}/"
        f"{embedder.name}/{embedder.dimension}"
    )


def index_name(profile: str, embedder: Embedder) -> str:
    """Nom de collection (Qdrant) et d'index (OpenSearch) : un par cas d'usage
    et par clé d'index. Un changement d'embedder crée un nouvel index au lieu
    de mélanger des vecteurs incomparables."""
    digest = hashlib.sha256(index_key(embedder).encode()).hexdigest()[:8]
    return f"luciole_{profile}_v{INDEX_SCHEMA_VERSION}_{digest}"
