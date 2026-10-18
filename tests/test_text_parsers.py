"""Parsers texte, Markdown et transcriptions, sur le corpus de réunions fictives."""

from pathlib import Path

import pytest

from luciole.ingestion.base import ParserRegistry
from luciole.ingestion.text_parsers import (
    MarkdownParser,
    TextParser,
    TranscriptParser,
    default_text_registry,
)

CORPUS = Path(__file__).resolve().parent / "corpus" / "ventes"


def test_corpus_reunions_present():
    assert sorted(p.name for p in CORPUS.glob("*.transcript.md")) == [
        "atelier-dubois_reunion-1.transcript.md",
        "atelier-dubois_reunion-2.transcript.md",
        "atelier-dubois_reunion-3.transcript.md",
    ]


@pytest.mark.parametrize("n", [1, 2, 3])
def test_transcription_reunion(n):
    doc = TranscriptParser().parse(CORPUS / f"atelier-dubois_reunion-{n}.transcript.md")
    assert doc.doc_type == "transcript"
    # Le front matter alimente les métadonnées du cas d'usage « ventes ».
    assert doc.metadata["client"] == "Atelier Dubois (fictif)"
    assert doc.metadata["numero_reunion"] == n
    assert doc.metadata["commercial"] == "Camille Martin"
    turns = [x for x in doc.nodes if x.node_type == "speaker_turn"]
    assert len(turns) >= 6
    assert {t.attributes["speaker"] for t in turns} >= {"Camille Martin"}
    assert all(t.attributes.get("timestamp") for t in turns), "chaque tour est horodaté"
    assert doc.metadata["speaker_count"] >= 2
    assert doc.pages[0].quality == 1.0


def test_transcription_tour_sur_plusieurs_lignes(tmp_path):
    f = tmp_path / "x.transcript.md"
    f.write_text(
        "## Réunion\n\n[00:01] Camille Martin : Première ligne\nsuite de la phrase.\n\n"
        "Paul Dubois - Réponse courte.\nNote libre sans locuteur.\n",
        encoding="utf-8",
    )
    doc = TranscriptParser().parse(f)
    kinds = [n.node_type for n in doc.nodes]
    assert kinds == ["heading", "speaker_turn", "speaker_turn"]
    assert doc.nodes[1].text == "Première ligne suite de la phrase."
    assert doc.nodes[1].section_path == ("Réunion",)
    assert doc.nodes[2].attributes == {"speaker": "Paul Dubois"}
    assert doc.nodes[2].text == "Réponse courte. Note libre sans locuteur."


def test_markdown_structure(tmp_path):
    f = tmp_path / "note.md"
    f.write_text(
        "---\ntitle: Note\nprojet: Parc\n---\n# Titre\n\nUn paragraphe\nsur deux lignes.\n\n"
        "## Section\n\n- a\n- b\n\n| c1 | c2 |\n|---|---|\n| 1 | 2 |\n\n```\ncode\n```\n",
        encoding="utf-8",
    )
    doc = MarkdownParser().parse(f)
    assert doc.metadata == {"title": "Note", "projet": "Parc"}
    types = [n.node_type for n in doc.nodes]
    assert types == ["title", "paragraph", "heading", "list_item", "list_item", "table", "code"]
    assert doc.nodes[1].text == "Un paragraphe sur deux lignes."
    table = doc.nodes[5]
    assert table.attributes == {"rows": 3} and table.section_path == ("Titre", "Section")
    assert doc.nodes[6].text == "code"


def test_markdown_front_matter_invalide_ignore(tmp_path):
    f = tmp_path / "x.md"
    f.write_text("---\n: pas du yaml\n---\n# T\n", encoding="utf-8")
    doc = MarkdownParser().parse(f)
    assert doc.metadata == {}
    assert doc.nodes[0].node_type == "title"


def test_texte_brut_et_cp1252(tmp_path):
    f = tmp_path / "x.txt"
    f.write_bytes("Premier bloc.\n\nDeuxième bloc à l'été.".encode("cp1252"))
    doc = TextParser().parse(f)
    assert [n.text for n in doc.nodes] == ["Premier bloc.", "Deuxième bloc à l'été."]


def test_fichier_vide_donne_page_sans_methode(tmp_path):
    f = tmp_path / "vide.txt"
    f.write_text("  \n", encoding="utf-8")
    doc = TextParser().parse(f)
    assert doc.nodes == [] and doc.pages[0].method is None and doc.pages[0].quality == 0


def test_registre_suffixes_composes():
    reg = default_text_registry()
    assert type(reg.parser_for("a.transcript.md")).__name__ == "TranscriptParser"
    assert type(reg.parser_for("a.md")).__name__ == "MarkdownParser"
    assert type(reg.parser_for("A.TXT")).__name__ == "TextParser"
    assert not reg.supports("a.pdf")
    with pytest.raises(ValueError, match="non pris en charge"):
        reg.parser_for("a.pdf")
    with pytest.raises(FileNotFoundError):
        reg.parse("inexistant.md")


def test_registre_simple_refuse_les_doublons():
    with pytest.raises(ValueError, match="déjà prise"):
        ParserRegistry([TextParser(), TextParser()])
