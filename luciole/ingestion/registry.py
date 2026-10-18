"""Registre par défaut : tous les parsers disponibles, suffixes composés compris."""

from __future__ import annotations

from luciole.ingestion.ocr import OcrEngine
from luciole.ingestion.pdf import PdfParser
from luciole.ingestion.text_parsers import (
    MarkdownParser,
    TextParser,
    TranscriptParser,
    _CompoundSuffixRegistry,
)


def default_registry(*, ocr: OcrEngine | None = None, **pdf_options) -> _CompoundSuffixRegistry:
    """Parsers texte, Markdown, transcriptions et PDF. ``pdf_options`` sont
    les paramètres de ``PdfParser`` (seuils, DPI), lus du profil à l'étape 3."""
    return _CompoundSuffixRegistry(
        [TranscriptParser(), MarkdownParser(), TextParser(), PdfParser(ocr=ocr, **pdf_options)]
    )
