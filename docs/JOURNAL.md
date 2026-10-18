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
- Résultat : _en cours_.
