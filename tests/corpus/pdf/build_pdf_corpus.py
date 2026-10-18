"""Corpus PDF synthétique pour les tests du parser PDF.

Les fichiers sont générés (pas versionnés) : les tests les construisent dans
un dossier temporaire, et Damien peut les produire pour les ouvrir :

    uv run python tests/corpus/pdf/build_pdf_corpus.py <dossier_de_sortie>

Chaque fichier isole un cas que V5 traitait mal ou globalement :

- ``simple.pdf`` : 3 pages A4, en-tête et pied de page répétés, titres.
- ``a3_dense.pdf`` : 2 pages A3 paysage à mise en page dense (le cas des
  études d'impact où pymupdf4llm perdait ~65 % du texte).
- ``scanned.pdf`` : 1 page image sans texte natif (OCR requis).
- ``tables.pdf`` : 1 page avec un tableau tracé.
- ``vector_plan.pdf`` : 1 page de plan vectoriel (> 1000 tracés, 3 étiquettes).
- ``mixed.pdf`` : texte, puis page scannée, puis texte : le routage doit se
  décider page par page.
- ``two_up_rotated.pdf`` : 4 pages portrait **tournées de 90°** et imprimées
  « deux pages côte à côte » (le format des études d'impact wpd), avec un
  en-tête et un pied par demi-page, et un verso blanc.
- ``dense_table.pdf`` : 1 page avec un tableau de 600 cellules tracées (50 × 12) :
  plus de 500 tracés, mais c'est un tableau à lire en Markdown, pas un plan.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

A4 = pymupdf.paper_rect("a4")
A3_LANDSCAPE = pymupdf.Rect(0, 0, 1191, 842)

LOREM = (
    "Le projet s'inscrit dans un contexte de plaine agricole ouverte. "
    "Les inventaires menés sur un cycle biologique complet ont permis de "
    "caractériser les enjeux relatifs aux chiroptères et à l'avifaune. "
    "Les mesures d'évitement et de réduction sont décrites au chapitre suivant. "
)


def _header_footer(page: pymupdf.Page, number: int, total: int) -> None:
    w = page.rect.width
    page.insert_text((40, 30), "Projet Vent du Nord — Tome 4 — Étude d'impact", fontsize=8)
    page.insert_text((w - 120, page.rect.height - 25), f"Page {number} / {total}", fontsize=8)


def build_simple(path: Path) -> None:
    doc = pymupdf.open()
    for i in range(3):
        page = doc.new_page(width=A4.width, height=A4.height)
        _header_footer(page, i + 1, 3)
        if i == 0:
            page.insert_text((50, 90), "Résumé non technique", fontsize=20)
        page.insert_text((50, 130), f"{i + 1}. Chapitre {i + 1}", fontsize=14)
        page.insert_textbox(pymupdf.Rect(50, 150, 545, 500), LOREM * 4, fontsize=10)
    doc.save(path)
    doc.close()


def build_a3_dense(path: Path) -> None:
    doc = pymupdf.open()
    for p in range(2):
        page = doc.new_page(width=A3_LANDSCAPE.width, height=A3_LANDSCAPE.height)
        _header_footer(page, p + 1, 2)
        page.insert_text((40, 60), f"Carte {p + 1} — Enjeux écologiques", fontsize=16)
        cols, rows = 4, 3
        cw, rh = 270, 230
        for r in range(rows):
            for c in range(cols):
                x0, y0 = 40 + c * (cw + 15), 90 + r * (rh + 15)
                rect = pymupdf.Rect(x0, y0, x0 + cw, y0 + rh)
                page.draw_rect(rect, color=(0.6, 0.6, 0.6), width=0.5)
                page.insert_textbox(
                    rect + (4, 4, -4, -4),
                    f"Encart {r * cols + c + 1}. " + LOREM * 2,
                    fontsize=7,
                )
    doc.save(path)
    doc.close()


def _text_page_as_image(text: str, dpi: int = 150) -> pymupdf.Pixmap:
    tmp = pymupdf.open()
    page = tmp.new_page(width=A4.width, height=A4.height)
    page.insert_textbox(pymupdf.Rect(60, 80, 535, 700), text, fontsize=13)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72))
    tmp.close()
    return pix


def _add_scanned_page(doc: pymupdf.Document, text: str) -> None:
    pix = _text_page_as_image(text)
    page = doc.new_page(width=A4.width, height=A4.height)
    page.insert_image(page.rect, pixmap=pix)


def build_scanned(path: Path) -> None:
    doc = pymupdf.open()
    _add_scanned_page(doc, "Document scanné.\n\n" + LOREM * 3)
    doc.save(path)
    doc.close()


def build_tables(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=A4.width, height=A4.height)
    page.insert_text((50, 60), "Tableau 12 : espèces contactées", fontsize=12)
    headers = ["Espèce", "Statut", "Effectif", "Enjeu"]
    rows = [
        ["Pipistrelle commune", "Protégée", "412", "Fort"],
        ["Noctule de Leisler", "Protégée", "37", "Fort"],
        ["Alouette des champs", "Déclin", "18", "Modéré"],
        ["Busard cendré", "Protégée", "2", "Très fort"],
        ["Faucon crécerelle", "Commune", "6", "Faible"],
    ]
    x0, y0, cw, rh = 50, 90, 120, 22
    for r, cells in enumerate([headers, *rows]):
        for c, cell in enumerate(cells):
            rect = pymupdf.Rect(x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh)
            page.draw_rect(rect, color=(0, 0, 0), width=0.6)
            page.insert_textbox(rect + (3, 3, -3, -3), cell, fontsize=9)
    page.insert_textbox(
        pymupdf.Rect(50, 260, 545, 400), "Source : inventaires 2025. " + LOREM, fontsize=10
    )
    doc.save(path)
    doc.close()


def build_vector_plan(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=A3_LANDSCAPE.width, height=A3_LANDSCAPE.height)
    shape = page.new_shape()
    for i in range(1200):
        x = 40 + (i * 37) % 1100
        y = 60 + (i * 53) % 740
        shape.draw_line((x, y), (x + 15, y + 9))
        shape.finish(color=(0.2, 0.2, 0.2), width=0.3)
    shape.commit()
    page.insert_text((60, 50), "Plan de masse", fontsize=14)
    page.insert_text((400, 420), "E3", fontsize=9)
    page.insert_text((700, 300), "Poste de livraison", fontsize=9)
    doc.save(path)
    doc.close()


def build_mixed(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=A4.width, height=A4.height)
    page.insert_text((50, 80), "1. Avant-propos", fontsize=14)
    page.insert_textbox(pymupdf.Rect(50, 100, 545, 400), LOREM * 3, fontsize=10)
    _add_scanned_page(doc, "Annexe scannée : arrêté préfectoral.\n\n" + LOREM * 2)
    page = doc.new_page(width=A4.width, height=A4.height)
    page.insert_text((50, 80), "3. Conclusion", fontsize=14)
    page.insert_textbox(pymupdf.Rect(50, 100, 545, 400), LOREM * 3, fontsize=10)
    doc.save(path)
    doc.close()


TWO_UP_LEFT = ["Gauche : contexte du site.", "Le site est en plaine agricole."]
TWO_UP_RIGHT = ["Droite : méthode de mesure.", "Six points fixes ont été instrumentés."]


def _put_rotated(page: pymupdf.Page, x: float, y: float, text: str, size: float = 10) -> None:
    """Écrit ``text`` à la position (x, y) de la page **affichée** d'une page
    tournée de 90° (les coordonnées d'écriture sont celles de la page non tournée)."""
    page.insert_text(pymupdf.Point(x, y) * page.derotation_matrix, text, fontsize=size, rotate=90)


def build_two_up_rotated(path: Path) -> None:
    """Page portrait A3 (842 × 1191) tournée de 90° : affichée en paysage, elle
    porte deux demi-pages (gauche, droite) avec chacune un en-tête de chapitre
    en haut et un pied « Projet … n/47 » en bas. La page 2 est un verso blanc."""
    doc = pymupdf.open()
    numbers = iter(range(10, 20))
    for p in range(4):
        page = doc.new_page(width=842, height=1191)
        page.set_rotation(90)
        if p == 1:
            continue  # verso blanc : ni texte, ni image, ni tracé

        for half, x in enumerate((40, 640)):
            _put_rotated(page, x, 30, "ETAT ACOUSTIQUE INITIAL", 8)
            _put_rotated(
                page, x, 820, f"Projet Vent du Nord - Rapport acoustique    {next(numbers)}/47", 7
            )
            lines = TWO_UP_LEFT if half == 0 else TWO_UP_RIGHT
            for i, line in enumerate(lines):
                _put_rotated(page, x, 120 + 40 * i + 15 * half, f"{line} (page {p + 1})")
            # Cellule de tableau en bas de demi-page : ne doit pas devenir un pied.
            _put_rotated(page, x + 200, 790, "0,0", 7)
    doc.save(path)
    doc.close()


def build_dense_table(path: Path) -> None:
    """50 lignes × 12 colonnes = 600 cellules tracées, lisibles (6,5 pt)."""
    doc = pymupdf.open()
    page = doc.new_page(width=A4.width, height=A4.height)
    page.insert_text((50, 50), "Tableau 7 : niveaux résiduels par vitesse de vent", fontsize=11)
    x0, y0, cw, rh = 40, 70, 43, 14
    for r in range(50):
        if r == 0:
            cells = ["Vitesse", *(f"PF{c}" for c in range(1, 12))]
        else:
            cells = [f"{r} m/s", *(f"{30 + (r * c) % 25},5" for c in range(1, 12))]
        for c, cell in enumerate(cells):
            rect = pymupdf.Rect(x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh)
            page.draw_rect(rect, color=(0, 0, 0), width=0.3)
            page.insert_text((rect.x0 + 2, rect.y1 - 4), cell, fontsize=6.5)
    doc.save(path)
    doc.close()


BUILDERS = {
    "simple.pdf": build_simple,
    "a3_dense.pdf": build_a3_dense,
    "scanned.pdf": build_scanned,
    "tables.pdf": build_tables,
    "vector_plan.pdf": build_vector_plan,
    "mixed.pdf": build_mixed,
    "two_up_rotated.pdf": build_two_up_rotated,
    "dense_table.pdf": build_dense_table,
}


def build_all(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, builder in BUILDERS.items():
        p = out_dir / name
        builder(p)
        paths[name] = p
    return paths


if __name__ == "__main__":  # pragma: no cover
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("corpus_pdf")
    for name, p in build_all(target).items():
        print(f"{name:16s} {p.stat().st_size / 1024:7.1f} Ko")
