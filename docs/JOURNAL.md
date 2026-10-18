# Journal des étapes

Une entrée par étape : ce qui a été fait, ce qui a été mesuré, ce qui a été décidé, ce qui reste. Rédigé par Claude à la fermeture de l'étape, relu par Damien.

## Étape 0 — Mise en place (ouverte le 2026-10-18)

- Dépôt créé par Damien, PR d'étape 0 ouverte par Claude.
- Porte de passage : `uv run pytest` vert dans GitHub Actions et dans PowerShell sur la 3080 Ti.
- Résultat (2026-10-18) : CI GitHub Actions verte (ruff + 29 tests) sur la PR #1 ; sur la RTX 3080 Ti de Damien (Windows 11, Python 3.12.13) : `DEV.ps1 -Action install` OK, `test` 29 passed en 0,72 s, `smoke` détecte x86_64, 12 Go et suggère `rtx3080ti`. Porte franchie.
- Correctif au passage : `DEV.ps1` enregistré en UTF-8 avec BOM, sinon PowerShell 5 affiche les accents en « dÃ© ».

## Étape 1 — Modèle documentaire et parser PDF (ouverte le 2026-10-18)

- Porte de passage : tests verts sur le corpus de test (PDF variés + comptes rendus de réunion fictifs) ; sur une page A3 dense, texte récupéré ≥ V5.
- Découpage prévu : PR 1a `DocumentNode` + parsers texte, Markdown, transcriptions + corpus réunions fictives ; PR 1b parser PDF page par page (routage natif / Markdown / OCR, score qualité, OCR à sa page) + corpus PDF synthétique ; PR 1c DOCX et autres formats bureautiques.
- 2026-10-18, PR 1d : premier passage du parser PDF sur trois études d'impact publiques (202 pages A3 paysage, « deux pages côte à côte »). Mesures : pymupdf4llm rend les tableaux jusqu'à ~2000 tracés par page en moins de 3 s (rendement ≥ 100 % du natif) ; les cartes font 7000 à 15000 tracés et n'ont pas de tableau. Défauts trouvés et corrigés : coordonnées des blocs non tournées (PDF portrait tourné de 90° : ordre de lecture faux, en-têtes ratés), versos blancs comptés « à OCR » (11 pages sur 202), seuil vectoriel de 500 trop bas (16 pages de tableaux lues en natif), détection d'en-têtes limitée à un bloc par bord. Après correction : 0 page à OCR à tort, 6 pages en natif (toutes des cartes), 73 à 221 en-têtes/pieds typés par document. Les études restent hors du dépôt ; le corpus synthétique reproduit les cas (`two_up_rotated.pdf`, `dense_table.pdf`). Sur les doubles pages, l'ordre gauche puis droite est garanti en texte natif (tri par colonne) et vérifié sur une vraie page en Markdown (pymupdf4llm gère les colonnes ; découper la page en deux avant lui ne marche pas : `margins` et `set_cropbox` sont ignorés par pymupdf4llm 1.28, `show_pdf_page` brouille le texte). Les balises HTML laissées par pymupdf4llm (`<!-- Start of picture text -->`, `<br>`, `<mark>`) sont retirées. À traiter plus tard : numéros de page imprimés (« 14/47 ») différents des numéros PDF (étape 2, citations), cellules de tableaux fusionnées mal rendues.
- Résultat : _en cours_.
