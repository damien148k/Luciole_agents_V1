# Décisions techniques

Une ligne par choix : ce qu'on a décidé, pourquoi, quand. On ne rediscute pas un choix noté sans nouvelle mesure ; si on change d'avis, on ajoute une ligne, on ne réécrit pas l'ancienne.

| Date | Décision | Raison |
|---|---|---|
| 2026-10-18 | Nouveau dépôt, pas une évolution de V5 | V5 porte trois générations de commentaires (V3/V4/V5), un README faux et des défauts structurels (voir `cadrage/analyse-v5.md`) |
| 2026-10-18 | Le cas d'usage est l'objet central (`configs/profiles/<cas>/`) ; le RAG est un outil parmi d'autres | Deux cas d'usage connus (MRAe, commerciaux) ne partagent ni les documents, ni les métadonnées, ni la forme de la boucle |
| 2026-10-18 | Les boucles sont des machines à états en code, décrites en YAML, aux règles déterministes | Post-mortem Belacom : un LLM qui pilote le flux de contrôle a systématiquement fait moins bien qu'un pipeline à étapes séparées |
| 2026-10-18 | Python 3.12 et `uv` avec `uv.lock` | Reproductibilité hors ligne ; casse du 7 septembre 2026 sur V5 quand pip a remplacé torch |
| 2026-10-18 | Profils matériels (`configs/hardware/`) et images Docker `amd64` + `arm64` | Trois machines cibles dont le GX10 en ARM64 |
| 2026-10-18 | Chunking en tokens du modèle d'embedding, parent-enfant | V5 comptait en caractères (1024 caractères ≈ 250 tokens, loin des 512 à 1024 tokens de la spec) |
| 2026-10-18 | `document_id` = hash du chemin relatif et du contenu | V5 utilisait le nom de fichier seul : collisions entre dossiers |
| 2026-10-18 | Qdrant ≥ 1.12 avec vecteurs nommés dès le départ ; OpenSearch gardé pour BM25 | Permet multi-dense puis sparse sans changer de schéma ; BM25 reste la base de comparaison |
| 2026-10-18 | Magasin structuré local (SQLite ou DuckDB) pour les métadonnées | Le cas commerciaux filtre et agrège sur des dizaines de dossiers avant de lire |
