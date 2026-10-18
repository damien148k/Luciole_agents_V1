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
- Fait : PR #2 (1a : `DocumentNode`, `ParsedDocument`, `NodeBuilder`, parsers texte / Markdown / transcriptions, corpus fictif `tests/corpus/ventes`), #3 (1b : parser PDF page par page, contrat `OcrEngine` + EasyOCR en extra optionnel, corpus PDF synthétique), #4 (1c : DOCX, PPTX, XLSX, HTML, EML, registre par défaut), #5 (1d : réglages sur études réelles), #6 (rattrapage : #3 à #5 avaient été fusionnées dans leurs branches intermédiaires, pas dans `main`). 14 formats lus, 75 tests, 24 s en local.
- Mesuré : porte « A3 dense » tenue par `test_a3_dense_ne_perd_pas_le_texte` (≥ 95 % du texte natif récupéré, V5 en perdait ~65 %) ; sur 202 pages réelles, aucune page perdue, 6 pages de cartes en natif, 0 OCR demandé à tort, 73 à 221 en-têtes/pieds écartés par document. CI GitHub Actions verte sur `main` (7767d7d).
- Décidé : voir `DECISIONS.md` (seuils PDF, contrat OCR, une PR à la fois sur `main`).
- Reporté : OCR réel (moteur EasyOCR branché à l'étape 3 avec le profil matériel) ; numéros de page imprimés (« 14/47 ») à retrouver pour les citations (étape 2) ; cellules fusionnées des tableaux mal rendues par pymupdf4llm ; vérification locale de `main` par Damien (75 tests) attendue mais non bloquante, la CI faisant foi pour une étape sans GPU.
- Résultat (2026-10-18) : **porte franchie.**

## Étape 2 — Dossiers, métadonnées typées, chunking (ouverte le 2026-10-18)

- Porte de passage : deux fichiers de même nom coexistent ; les champs d'une réunion fictive sont extraits et validés ; aucun chunk ne dépasse la limite du tokenizer.
- Découpage prévu : PR 2a dossiers, `document_id`, métadonnées typées par règles, magasin structuré SQLite, corpus fictif à deux dossiers ; PR 2b chunking en tokens BGE-M3, parent-enfant, garde de longueur, numéros de page imprimés (« 14/47 »).
- 2026-10-18, PR 2a : les deux premiers points de la porte sont couverts par `test_deux_fichiers_de_meme_nom_coexistent` et `test_champs_d_une_reunion_fictive_extraits_et_valides` ; 151 tests. Le troisième point attend la PR 2b.
