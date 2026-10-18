"""Registre par défaut : tous les parsers disponibles, suffixes composés compris."""

from __future__ import annotations

from luciole.ingestion.ocr import OcrEngine
from luciole.ingestion.office import DocxParser, PptxParser, XlsxParser
from luciole.ingestion.pdf import PdfParser
from luciole.ingestion.text_parsers import (
    MarkdownParser,
    TextParser,
    TranscriptParser,
    _CompoundSuffixRegistry,
)
from luciole.ingestion.web_mail import EmlParser, HtmlParser


def default_registry(*, ocr: OcrEngine | None = None, **pdf_options) -> _CompoundSuffixRegistry:
    """Parsers texte, Markdown, transcriptions, PDF, DOCX, PPTX, XLSX, HTML et
    EML. ``pdf_options`` sont les paramètres de ``PdfParser`` (seuils, DPI),
    lus du profil à l'étape 3."""
    return _CompoundSuffixRegistry(
        [
            TranscriptParser(),
            MarkdownParser(),
            TextParser(),
            PdfParser(ocr=ocr, **pdf_options),
            DocxParser(),
            PptxParser(),
            XlsxParser(),
            HtmlParser(),
            EmlParser(),
        ]
    )
