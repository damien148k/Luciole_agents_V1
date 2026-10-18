"""Versions de l'application et de ses contrats internes.

V5 n'enregistrait aucune version : impossible de savoir quel schéma d'index
ou quel parser avait produit un chunk. Ici chaque chunk, chaque trace et la
route ``/api/health`` portent ces quatre numéros. Changer l'un d'eux est une
décision notée dans ``docs/DECISIONS.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

#: Version de l'application (suit ``pyproject.toml``).
APP_VERSION = "1.0.0a0"
#: Schéma des collections Qdrant et des index OpenSearch. À incrémenter dès
#: qu'un champ de payload, un vecteur nommé ou un analyseur change : les
#: index existants doivent alors être reconstruits.
INDEX_SCHEMA_VERSION = 1
#: Comportement des parsers (DocumentNode, découpage, OCR). À incrémenter
#: quand le même fichier produirait des chunks différents.
PARSER_VERSION = 1
#: Modèle d'embedding, préfixes et vecteurs nommés produits.
EMBEDDING_PROFILE_VERSION = 1


@dataclass(frozen=True)
class Versions:
    """Jeu de versions attaché à un chunk, une trace ou un health check."""

    app: str = APP_VERSION
    index_schema: int = INDEX_SCHEMA_VERSION
    parser: int = PARSER_VERSION
    embedding_profile: int = EMBEDDING_PROFILE_VERSION

    def as_dict(self) -> dict:
        return asdict(self)


def current() -> Versions:
    """Versions en vigueur dans ce processus."""
    return Versions()
