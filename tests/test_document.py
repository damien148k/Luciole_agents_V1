"""Modèle documentaire : invariants, hiérarchie des sections, projections."""

import pytest
from pydantic import ValidationError

from luciole.core.document import BBox, DocumentNode, NodeBuilder, PageInfo, ParsedDocument


def _doc(nodes, pages=None):
    return ParsedDocument(
        file_name="t.pdf", source_path="t.pdf", doc_type="pdf", nodes=nodes, pages=pages or []
    )


def test_builder_tient_la_hierarchie_des_sections():
    b = NodeBuilder()
    b.add("title", "Étude d'impact", page=1)
    b.add("heading", "1. Milieu naturel", page=2, level=2)
    p1 = b.add("paragraph", "Inventaire des chiroptères.", page=2)
    b.add("heading", "1.1 Chiroptères", page=3, level=3)
    p2 = b.add("paragraph", "Dix espèces contactées.", page=3)
    h2 = b.add("heading", "2. Paysage", page=4, level=2)
    p3 = b.add("paragraph", "Covisibilités.", page=4)

    assert p1.section_path == ("Étude d'impact", "1. Milieu naturel")
    assert p2.section_path == ("Étude d'impact", "1. Milieu naturel", "1.1 Chiroptères")
    # Un titre de niveau 2 referme le 1.1 (niveau 3) et le 1 (niveau 2).
    assert h2.section_path == ("Étude d'impact",)
    assert p3.section_path == ("Étude d'impact", "2. Paysage")
    assert p3.parent_id == h2.node_id
    doc = _doc(b.nodes())
    assert [n.text for n in doc.children_of(h2.node_id)] == ["Covisibilités."]
    assert len(doc.headings()) == 4


def test_reading_order_strictement_croissant():
    a = DocumentNode(node_id="a", node_type="paragraph", text="x", reading_order=1)
    b = DocumentNode(node_id="b", node_type="paragraph", text="y", reading_order=1)
    with pytest.raises(ValidationError, match="reading_order"):
        _doc([a, b])


def test_parent_inconnu_refuse():
    n = DocumentNode(node_id="a", parent_id="zz", node_type="paragraph", text="x", reading_order=0)
    with pytest.raises(ValidationError, match="parent"):
        _doc([n])


def test_pages_coherentes():
    n = DocumentNode(node_id="a", node_type="paragraph", text="x", reading_order=0, page_start=3)
    assert n.page_end == 3
    with pytest.raises(ValidationError):
        DocumentNode(
            node_id="b", node_type="paragraph", text="x", reading_order=0, page_start=3, page_end=2
        )
    with pytest.raises(ValidationError, match="page_end sans page_start"):
        DocumentNode(node_id="c", node_type="paragraph", text="x", reading_order=0, page_end=2)


def test_bbox_ordonnee():
    with pytest.raises(ValidationError):
        BBox(x0=10, y0=0, x1=5, y1=5)


def test_nodes_on_page_et_pages_uniques():
    b = NodeBuilder()
    b.add("paragraph", "p1", page=1)
    b.add("table", "| a |", page=2, page_end=3)
    b.add("paragraph", "p4", page=4)
    doc = _doc(b.nodes(), pages=[PageInfo(number=i) for i in range(1, 5)])
    assert [n.text for n in doc.nodes_on_page(3)] == ["| a |"]
    assert doc.page_count == 4
    with pytest.raises(ValidationError, match="pages"):
        _doc([], pages=[PageInfo(number=2), PageInfo(number=1)])


def test_projections_excluent_entetes_et_pieds():
    b = NodeBuilder()
    b.add("header", "Tome 4 — Étude d'impact", page=1)
    b.add("title", "Résumé", page=1)
    b.add("paragraph", "Texte.", page=1)
    b.add("list_item", "un point", page=1)
    b.add(
        "speaker_turn",
        "Bonjour.",
        page=1,
        attributes={"speaker": "C. Martin", "timestamp": "00:01"},
    )
    b.add("footer", "page 1/120", page=1)
    doc = _doc(b.nodes())
    assert "Tome 4" not in doc.to_text() and "page 1/120" not in doc.to_text()
    md = doc.to_markdown()
    assert md.startswith("# Résumé")
    assert "- un point" in md
    assert "**[00:01] C. Martin :** Bonjour." in md
    assert doc.to_text(types=["paragraph"]) == "Texte."
