# Corpus de test

Documents **fictifs et non confidentiels** utilisés par les tests. Aucun nom de client réel, aucune donnée réelle.

- `ventes/` : trois réunions d'un dossier commercial inventé (client « Atelier Dubois », produit « Luciole Pro », commerciale Camille Martin), au format transcription (`*.transcript.md`, front matter YAML + tours de parole horodatés). Le dossier est **perdu** à la réunion 3 : il sert à tester l'analyse « pourquoi pas de vente ».
- `pdf/` (étape 1b) : PDF synthétiques générés par `build_pdf_corpus.py` (texte simple, A3 paysage dense, page scannée, tableaux, plan vectoriel).
