"""Contrat des parsers et aiguillage par extension.

Un parser lit un fichier et renvoie un ``ParsedDocument`` (nœuds, pages,
métadonnées). Il ne découpe pas, n'encode pas, n'indexe pas. Les parsers
lourds (PDF, DOCX) sont enregistrés par les étapes suivantes ; ce module ne
dépend d'aucune bibliothèque externe.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from luciole.core.document import ParsedDocument


class Parser(ABC):
    """Un parser par famille de formats."""

    #: Extensions prises en charge, en minuscules et avec le point.
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def parse(self, path: str | Path) -> ParsedDocument:
        """Lit ``path`` et renvoie le document structuré. Lève ``FileNotFoundError``
        si le fichier manque, ``ValueError`` si le contenu est illisible."""


class ParserRegistry:
    """Choisit le parser d'un fichier d'après son extension."""

    def __init__(self, parsers: list[Parser] | None = None) -> None:
        self._by_ext: dict[str, Parser] = {}
        for p in parsers or []:
            self.register(p)

    def register(self, parser: Parser) -> None:
        for ext in parser.extensions:
            ext = ext.lower()
            if ext in self._by_ext:
                raise ValueError(
                    f"extension {ext} déjà prise par {type(self._by_ext[ext]).__name__}"
                )
            self._by_ext[ext] = parser

    def supported_extensions(self) -> list[str]:
        return sorted(self._by_ext)

    def supports(self, path: str | Path) -> bool:
        return Path(path).suffix.lower() in self._by_ext

    def parser_for(self, path: str | Path) -> Parser:
        ext = Path(path).suffix.lower()
        try:
            return self._by_ext[ext]
        except KeyError:
            raise ValueError(
                f"format non pris en charge : {ext or '(sans extension)'} ; "
                f"formats connus : {', '.join(self.supported_extensions())}"
            ) from None

    def parse(self, path: str | Path) -> ParsedDocument:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return self.parser_for(path).parse(path)
