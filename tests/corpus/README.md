# Corpus de test

Documents **fictifs et non confidentiels** utilisés par les tests. Aucun nom de client réel, aucune donnée réelle.

- `ventes/` : une racine d'ingestion, un sous-répertoire par dossier commercial inventé, réunions au format transcription (`*.transcript.md`, front matter YAML + tours de parole horodatés). Les deux dossiers ont un fichier `reunion-1.transcript.md` : c'est voulu (deux fichiers de même nom doivent coexister).
  - `atelier-dubois/` : trois réunions (commerciale Camille Martin), toutes les métadonnées dans le front matter, sans `dossier.yaml` ; l'issue passe de « en cours » à **perdu** à la réunion 3 (budget) : sert à l'analyse « pourquoi pas de vente » et à la règle « le document le plus récent l'emporte ».
  - `cabinet-roche/` : deux réunions (commercial Julien Bernard), dossier **gagné** à la réunion 2 (démonstration sur un vrai dossier, retour sur investissement chiffré) ; champs du dossier dans `dossier.yaml`, numéro de réunion lu dans le nom de fichier, une date au format JJ/MM/AAAA.
- `pdf/` (étape 1b) : PDF synthétiques générés par `build_pdf_corpus.py` (texte simple, A3 paysage dense, page scannée, tableaux, plan vectoriel).
