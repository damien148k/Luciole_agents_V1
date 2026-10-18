"""Parser PDF page par page, sur le corpus synthétique."""

import sys
from pathlib import Path

import pymupdf
import pytest

from luciole.core.document import BBox
from luciole.ingestion.ocr import OcrLine, group_lines, render_page, to_points
from luciole.ingestion.pdf import PdfParser, _clean_markdown, _is_margin_line, _normalize
from luciole.ingestion.registry import default_registry

sys.path.insert(0, str(Path(__file__).resolve().parent / "corpus" / "pdf"))
from build_pdf_corpus import LOREM, TWO_UP_LEFT, TWO_UP_RIGHT, build_all  # noqa: E402


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> dict[str, Path]:
    return build_all(tmp_path_factory.mktemp("pdf"))


class FakeOcr:
    """Moteur factice : renvoie deux lignes en pixels, quelle que soit l'image."""

    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, image_png: bytes) -> list[OcrLine]:
        self.calls += 1
        return [
            OcrLine("Document scanné.", BBox(x0=250, y0=330, x1=900, y1=380), 0.95),
            OcrLine("Suite du texte reconnu.", BBox(x0=250, y0=390, x1=1200, y1=440), 0.85),
            OcrLine("Bloc plus bas.", BBox(x0=250, y0=900, x1=700, y1=950), 0.60),
        ]


# ---------------------------------------------------------------------------
# Routage par page
# ---------------------------------------------------------------------------
def test_simple_markdown_et_entetes_pieds(corpus):
    doc = PdfParser().parse(corpus["simple.pdf"])
    assert doc.metadata["page_count"] == 3
    assert [p.method for p in doc.pages] == ["markdown"] * 3
    assert all(p.quality > 0.8 and not p.needs_ocr for p in doc.pages)
    headers = [n for n in doc.nodes if n.node_type == "header"]
    footers = [n for n in doc.nodes if n.node_type == "footer"]
    assert len(headers) == 3 and len(footers) == 3
    assert "Vent du Nord" not in doc.to_text() and "Page 2 / 3" not in doc.to_text()
    assert "chiroptères" in doc.to_text()
    assert doc.headings(), "les titres de chapitre sont des nœuds heading"
    assert doc.nodes_on_page(2) and all(n.page_start == 2 for n in doc.nodes_on_page(2))


def test_a3_dense_ne_perd_pas_le_texte(corpus):
    """Porte de passage de l'étape 1 : sur une page A3 dense, le texte
    récupéré vaut au moins celui de l'extraction native (V5 en perdait ~65 %
    avant son garde-fou)."""
    parser = PdfParser()
    doc = parser.parse(corpus["a3_dense.pdf"])
    with pymupdf.open(corpus["a3_dense.pdf"]) as raw:
        native_chars = sum(len(p.get_text("text").strip()) for p in raw)
    recovered = len(doc.to_text())
    assert recovered >= 0.95 * native_chars, (recovered, native_chars)
    assert all(p.method in ("markdown", "native_text") for p in doc.pages)
    assert all(p.quality >= 0.5 for p in doc.pages)


def test_garde_fou_de_rendement_replie_sur_le_natif(corpus, monkeypatch):
    """Si pymupdf4llm rend une page presque vide, cette page seule repasse en natif."""
    parser = PdfParser()
    monkeypatch.setattr(PdfParser, "_markdown", staticmethod(lambda doc, i: "trop court"))
    doc = parser.parse(corpus["simple.pdf"])
    assert all(p.method == "native_text" and p.fallbacks == ["markdown"] for p in doc.pages)
    assert all("rendement markdown" in p.notes[0] for p in doc.pages)
    assert "chiroptères" in doc.to_text()
    assert all(n.bbox is not None for n in doc.nodes), "les blocs natifs ont une boîte"


def test_echec_pymupdf4llm_replie_sur_le_natif(corpus, monkeypatch):
    monkeypatch.setattr(PdfParser, "_markdown", staticmethod(lambda doc, i: None))
    doc = PdfParser().parse(corpus["simple.pdf"])
    assert all(p.method == "native_text" and "échec" in p.notes[0] for p in doc.pages)


def test_tableau_detecte(corpus):
    doc = PdfParser().parse(corpus["tables.pdf"])
    tables = [n for n in doc.nodes if n.node_type == "table"]
    assert tables, "pymupdf4llm doit rendre le tableau tracé en Markdown"
    assert tables[0].attributes["rows"] >= 5
    assert "Pipistrelle commune" in tables[0].text and "|" in tables[0].text


def test_plan_vectoriel_en_natif(corpus):
    doc = PdfParser(max_drawings_per_page=500).parse(corpus["vector_plan.pdf"])
    page = doc.pages[0]
    assert page.method == "native_text"
    assert any("vectorielle" in note for note in page.notes)
    assert {n.text for n in doc.nodes} >= {"Plan de masse", "E3", "Poste de livraison"}


def test_page_scannee_sans_ocr(corpus):
    doc = PdfParser().parse(corpus["scanned.pdf"])
    page = doc.pages[0]
    assert page.needs_ocr and page.method is None and page.quality == 0
    assert "aucun moteur OCR" in page.notes[0]
    assert doc.nodes == []
    assert doc.metadata["pages_need_ocr"] == 1


def test_page_scannee_avec_ocr(corpus):
    ocr = FakeOcr()
    doc = PdfParser(ocr=ocr, ocr_dpi=300).parse(corpus["scanned.pdf"])
    page = doc.pages[0]
    assert ocr.calls == 1
    assert page.method == "ocr" and page.needs_ocr
    assert page.ocr_confidence == pytest.approx(0.8, abs=1e-3)
    assert page.quality == page.ocr_confidence
    nodes = doc.nodes
    assert [n.method for n in nodes] == ["ocr", "ocr"], "lignes proches groupées, bloc bas séparé"
    assert nodes[0].text == "Document scanné. Suite du texte reconnu."
    assert nodes[0].confidence == pytest.approx(0.9)
    # Boîtes converties des pixels (300 dpi) vers les points : 250 px → 60 pt.
    assert nodes[0].bbox.x0 == pytest.approx(60.0)
    assert nodes[0].attributes["ocr_engine"] == "fake"
    assert nodes[0].page_start == 1


def test_routage_page_par_page(corpus):
    """Le point 5.4 de l'analyse V5 : une page scannée au milieu d'un document
    textuel ne change pas le traitement des autres pages, et son texte OCR
    reste à sa place dans l'ordre de lecture."""
    doc = PdfParser(ocr=FakeOcr()).parse(corpus["mixed.pdf"])
    assert [p.method for p in doc.pages] == ["markdown", "ocr", "markdown"]
    assert [p.needs_ocr for p in doc.pages] == [False, True, False]
    order = [(n.page_start, n.method) for n in doc.nodes]
    assert order == sorted(order, key=lambda t: t[0]), "les nœuds suivent l'ordre des pages"
    p2 = doc.nodes_on_page(2)
    assert p2 and all(n.method == "ocr" for n in p2)
    assert doc.metadata["pages_ocr"] == 1 and doc.metadata["pages_markdown"] == 2


def test_page_tournee_deux_pages_cote_a_cote(corpus):
    """Le format des études d'impact wpd : PDF portrait tourné de 90°, deux
    demi-pages par page. Les boîtes sont ramenées dans la page affichée, les
    en-têtes et pieds de chaque demi-page sont reconnus, et la demi-page de
    gauche se lit avant celle de droite."""
    doc = PdfParser(ocr=FakeOcr()).parse(corpus["two_up_rotated.pdf"])
    assert doc.metadata["page_count"] == 4
    assert [p.method for p in doc.pages] == ["markdown", None, "markdown", "markdown"]
    headers = [n for n in doc.nodes if n.node_type == "header"]
    footers = [n for n in doc.nodes if n.node_type == "footer"]
    assert len(headers) == 6 and len(footers) == 6, "deux par page écrite"
    cells = [n for n in doc.nodes if n.text == "0,0"]
    assert len(cells) == 6 and all(n.node_type == "paragraph" for n in cells), "cellule ≠ pied"
    text = doc.to_text()
    assert "ETAT ACOUSTIQUE INITIAL" not in text and "/47" not in text
    assert all(f"{t} (page 3)" in text for t in TWO_UP_LEFT + TWO_UP_RIGHT)


def test_page_tournee_en_natif_colonne_par_colonne(corpus, monkeypatch):
    """En texte natif (cartes, replis), les boîtes sont ramenées dans la page
    affichée et la demi-page de gauche se lit entièrement avant la droite."""
    monkeypatch.setattr(PdfParser, "_markdown", staticmethod(lambda doc, i: None))
    doc = PdfParser().parse(corpus["two_up_rotated.pdf"])
    assert doc.pages[0].method == "native_text"
    headers = [n for n in doc.nodes if n.node_type == "header"]
    footers = [n for n in doc.nodes if n.node_type == "footer"]
    assert len(headers) == 6 and len(footers) == 6
    assert all(n.bbox.y1 < 60 for n in headers), "boîtes dans le repère de la page affichée"
    assert all(n.bbox.y0 > 780 and n.bbox.x1 < 1191 for n in footers)
    body = [n.text for n in doc.nodes_on_page(1) if n.node_type == "paragraph" and n.text != "0,0"]
    assert body == [f"{t} (page 1)" for t in TWO_UP_LEFT + TWO_UP_RIGHT]


def test_page_blanche_n_est_pas_a_ocr(corpus):
    ocr = FakeOcr()
    doc = PdfParser(ocr=ocr).parse(corpus["two_up_rotated.pdf"])
    blank = doc.pages[1]
    assert blank.method is None and not blank.needs_ocr and blank.quality == 0
    assert blank.notes == ["page vide : ni texte, ni image, ni tracé"]
    assert ocr.calls == 0, "aucune page à OCR dans ce document"
    assert doc.metadata["pages_blank"] == 1 and doc.metadata["pages_need_ocr"] == 0
    assert doc.nodes_on_page(2) == []


def test_tableau_dense_reste_en_markdown(corpus):
    """600 cellules tracées dépassent l'ancien seuil de 500 tracés ; mesuré sur
    de vraies études : jusqu'à ~2000 tracés ce sont des tableaux que
    pymupdf4llm rend bien, le seuil « plan » est donc à 5000."""
    doc = PdfParser().parse(corpus["dense_table.pdf"])
    page = doc.pages[0]
    assert page.method == "markdown" and page.notes == []
    tables = [n for n in doc.nodes if n.node_type == "table"]
    assert tables and "PF3" in tables[0].text and "49 m/s" in doc.to_text()
    assert tables[0].attributes["rows"] >= 40
    with pymupdf.open(corpus["dense_table.pdf"]) as raw:
        assert len(raw[0].get_drawings()) > 500


def test_sans_ocr_le_texte_natif_d_une_page_pauvre_est_garde(tmp_path):
    pdf = tmp_path / "pauvre.pdf"
    d = pymupdf.open()
    page = d.new_page()
    page.insert_text((50, 80), "Figure 3", fontsize=10)
    d.save(pdf)
    d.close()
    doc = PdfParser().parse(pdf)
    assert doc.pages[0].needs_ocr and doc.pages[0].method == "native_text"
    assert [n.text for n in doc.nodes] == ["Figure 3"]


# ---------------------------------------------------------------------------
# Briques
# ---------------------------------------------------------------------------
def test_normalize_neutralise_les_numeros():
    assert _normalize("Page 3 / 120") == _normalize("Page 47 / 120")
    assert _normalize("Projet   Vent du Nord") == "projet vent du nord"
    assert _normalize("**Etude d’impact – tome 1**") == _normalize("Etude d’impact - tome 2")
    assert _normalize("14/47") != _normalize("0,0"), "numéro de page ≠ cellule de tableau"


def test_clean_markdown_retire_le_html_de_pymupdf4llm():
    md = (
        "<!-- Start of picture text -->\nPROJET EOLIEN<br>RAPPORT<br><!-- End of picture text -->"
        "\n\n**<mark>1 A</mark> RTICULATION**\n\n|Vitesse<br>1 m/s|PF1<br>31,5|\n|---|---|"
    )
    assert _clean_markdown(md) == (
        "\nPROJET EOLIEN\nRAPPORT\n\n\n**1 A RTICULATION**\n\n|Vitesse 1 m/s|PF1 31,5|\n|---|---|"
    ), "un <br> dans une ligne de tableau devient une espace, ailleurs un saut de ligne"


def test_is_margin_line():
    assert _is_margin_line(_normalize("ETAT ACOUSTIQUE INITIAL"))
    assert _is_margin_line(_normalize("14/47")) and _is_margin_line(_normalize("Page 3 sur 120"))
    assert _is_margin_line(_normalize("7"))
    assert not _is_margin_line(_normalize("0,0"))
    assert not _is_margin_line(_normalize("[135° ; 315°["))
    assert not _is_margin_line(_normalize("x" * 200))


def test_group_lines_et_conversion():
    a = OcrLine("a", BBox(x0=0, y0=0, x1=10, y1=10), 1.0)
    b = OcrLine("b", BBox(x0=0, y0=12, x1=10, y1=22), 1.0)
    c = OcrLine("c", BBox(x0=0, y0=60, x1=10, y1=70), 1.0)
    assert [[ln.text for ln in g] for g in group_lines([c, a, b])] == [["a", "b"], ["c"]]
    pts = to_points(BBox(x0=300, y0=0, x1=600, y1=72), scale=300 / 72)
    assert (pts.x0, pts.y0, pts.x1) == (72, 0, 144) and pts.y1 == pytest.approx(17.28)


def test_render_page_dimensions():
    d = pymupdf.open()
    page = d.new_page(width=200, height=100)
    r = render_page(page, dpi=144)
    assert (r.width, r.height) == (400, 200) and r.scale == 2.0 and r.png[:4] == b"\x89PNG"
    d.close()


def test_registre_par_defaut_connait_le_pdf(corpus):
    reg = default_registry()
    assert reg.supports("x.pdf") and reg.supports("x.transcript.md")
    doc = reg.parse(corpus["simple.pdf"])
    assert doc.doc_type == "pdf" and LOREM.split(".")[0] in doc.to_text()


# ---------------------------------------------------------------------------
# Numéros de page imprimés (étape 2)
# ---------------------------------------------------------------------------
def test_numeros_imprimes_double_page(corpus):
    doc = PdfParser().parse(corpus["two_up_rotated.pdf"])
    assert [[lb.display() for lb in p.labels] for p in doc.pages] == [
        ["10/47", "11/47"],
        [],
        ["12/47", "13/47"],
        ["14/47", "15/47"],
    ]
    assert [lb.side for lb in doc.pages[0].labels] == ["left", "right"]
    assert doc.metadata["pages_with_printed_label"] == 3


def test_numero_page_sur_total(corpus):
    doc = PdfParser().parse(corpus["simple.pdf"])
    assert [p.labels[0].display() for p in doc.pages] == ["1/3", "2/3", "3/3"]
    assert {p.labels[0].side for p in doc.pages} == {"page"}
    assert [n.node_type for n in doc.nodes].count("title") == 1, "un seul titre de document"


def _footer_pdf(path: Path, footers: list[str]) -> Path:
    doc = pymupdf.open()
    for text in footers:
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(pymupdf.Rect(50, 100, 545, 600), LOREM, fontsize=10)
        page.insert_text((50, 815), text, fontsize=8)
    doc.save(path)
    doc.close()
    return path


def test_numero_seul_en_pied_de_page(tmp_path):
    doc = PdfParser().parse(_footer_pdf(tmp_path / "t.pdf", ["12", "13", "14", "15"]))
    assert [p.labels[0].display() for p in doc.pages] == ["12", "13", "14", "15"]


@pytest.mark.parametrize(
    "footers",
    [
        ["Rapport – 01/06/2021"] * 4,  # une date n'est pas « page 6 sur 2021 »
        ["2021"] * 4,  # un nombre qui ne change pas n'est pas un numéro de page
        ["12", "", "", ""],  # un nombre isolé non plus
    ],
)
def test_pas_de_faux_numero_de_page(tmp_path, footers):
    doc = PdfParser().parse(_footer_pdf(tmp_path / "t.pdf", footers))
    assert all(p.labels == [] for p in doc.pages)
