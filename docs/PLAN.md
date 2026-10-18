# Luciole_agents_V1 — plan de refonte par étapes

Version 2 du 18 octobre 2026 (la version 1 du matin ne prévoyait ni les cas d'usage multiples, ni les trois machines). Ce fichier est lu par Damien et par Claude ; il est mis à jour à la fin de chaque étape.

## 1. Pourquoi repenser le projet, et comment

Luciole a été conçu en 2024 comme un RAG de qualité pour un cas unique (répondre aux avis MRAe). Il doit devenir une **plateforme de boucles agentiques par cas d'usage**, dont le RAG n'est qu'un des outils. Deux cas sont connus aujourd'hui, et ils ne se ressemblent pas :

| | Réponses aux avis MRAe | Assistant des commerciaux |
|---|---|---|
| Documents | Tomes d'étude d'impact (PDF longs, A3, cartes, tableaux) | Comptes rendus ou transcriptions de réunions client |
| Ce qui groupe les documents | Un **projet** (dossier déposé) | Un **dossier commercial** : client × produit × commercial, avec une suite de réunions 1, 2, 3… |
| Données structurées à connaître | Tome, chapitre, page | Commercial, client, produit, n° de réunion, date, issue (vente ou non) |
| Question type | « Répondre à cette remarque de l'autorité » | « Pourquoi ce dossier a-t-il abouti ? », « Qu'est-ce qui distingue les dossiers gagnés des perdus pour ce produit ? », « Préparer la réunion 3 » |
| Forme de la boucle | Analyse de la remarque → questions → recherche → gardes → synthèse → verdict | Chronologie du dossier → extraction de signaux par réunion → comparaison gagnés / perdus → synthèse |
| Ce dont la boucle a besoin | Recherche hybride fine, citations exactes | Filtrage et agrégation par métadonnées sur des dizaines de dossiers, puis lecture ciblée |

Ce que cela change dans la conception :

1. **Le cas d'usage devient l'objet central.** Un cas d'usage = un dossier `configs/profiles/<cas>/` qui décrit le schéma des entités (quels champs décrivent un document et un dossier), les règles d'ingestion, les réglages de recherche, la **boucle** (machine à états : nœuds, transitions, règles métier), les prompts et le jeu d'évaluation. Rien de tout cela ne vit dans le code.
2. **Un `Dossier` groupe les documents.** Chaque document appartient à un dossier (projet MRAe, dossier commercial) et porte des métadonnées typées extraites à l'ingestion, validées contre le schéma du cas d'usage. Pour les réunions, l'extraction des champs (commercial, client, produit, n°, date, issue) est faite par le LLM à sortie typée, puis vérifiée.
3. **Un magasin structuré à côté des index.** Les métadonnées vont aussi dans une base tabulaire locale (SQLite ou DuckDB) : la boucle peut filtrer, compter, grouper (tous les dossiers perdus du produit X) avant d'aller lire les textes. V5 avait déjà cette idée pour Excel (`sql_storage.py`).
4. **La boucle est une machine à états en code, décrite en YAML.** Les nœuds sont des outils typés fournis par le moteur : `retrieve` (hybride + rerank), `filter` et `aggregate` (magasin structuré), `extract` (LLM à sortie JSON validée), `compare`, `synthesize`, `verdict`, `notify`. Les transitions et l'arrêt sont décidés par des règles déterministes, jamais par le LLM (post-mortem Belacom). L'**éditeur graphique de boucles** édite ce YAML.
5. **Le pipeline MRAe (`query2`) devient le premier profil de boucle**, pas du code à part.

## 2. Trois machines, une seule base de code

| Machine | Rôle | Contraintes |
|---|---|---|
| RTX 3080 Ti, 12 Go, Windows | Développement et tests | LLM 7B à 14B quantifié, fenêtre 16k, lots d'embedding réduits |
| RTX A5000, 24 Go | Production MRAe | LLM 14B, fenêtre 32k |
| ASUS Ascent GX10 (puce GB10, mémoire unifiée 128 Go) | Production et gros modèles | **Processeur ARM64** : toutes les images Docker doivent exister en `linux/arm64`, et PyTorch/CUDA doit être installé pour aarch64. Modèles 70B possibles. |

Décisions : un fichier `configs/hardware/<machine>.yaml` par profil (modèle LLM, fenêtre, lots, device) choisi par `INSTALL.ps1` ou détecté ; images Docker construites en multi-architecture (`amd64` + `arm64`) dès l'étape 0 ; un **test de fumée** par machine (`DEV.ps1 -Action smoke`) qui vérifie GPU, modèles et conteneurs avant toute campagne. Qdrant, OpenSearch et Ollama existent en ARM64. Le point à vérifier tôt sur le GX10 est l'installation de PyTorch CUDA aarch64 et d'EasyOCR : c'est prévu dès l'étape 3, pas à la fin.

## 3. Ce que l'on garde de V5, ce que l'on refait

| Partie de V5 | Décision | Pourquoi |
|---|---|---|
| Documents de cadrage (`docs/cadrage/`) | **Repris** comme cahier des charges | Justes et réfléchis ; le code V5 ne les suivait pas encore |
| Logique mesurée sur le terrain (pagination, garde-fou de rendement PDF, en-têtes de tableaux, recouvrement sur les mots, gardes anti-esquive) | **Réécrite dans le nouveau modèle**, V5 relu comme référence | Des semaines de mesure (wpd, Beaumont Sud, Brissy) qu'on ne jette pas |
| Tests du watcher | **Portés** d'abord, code réécrit pour les faire passer | Les meilleurs tests du dépôt |
| Interfaces (10 000 lignes de HTML dans des chaînes Python) | **Refaites** en étape 6 sur une base propre | La moitié du code, dépend de tout le reste |
| `query2` | **Non repris en bloc** ; ses gardes deviennent des outils testés, son enchaînement devient le profil de boucle MRAe | Trop spécifique ; sa propre spec dit de ne pas le réutiliser |
| Scripts PowerShell | **Repris et adaptés** en étape 7 | Ils marchent |

## 4. Choix techniques par défaut (détail et dates dans `docs/DECISIONS.md`)

- Python 3.12, dépendances verrouillées par `uv` (`uv.lock`). Aucun `>=` libre, aucune image `latest`.
- Paquet `luciole/` : `core` (DocumentNode, Dossier, profils, versions, matériel), `ingestion`, `index`, `store` (magasin structuré), `retrieval`, `generation`, `agents` (boucles), `api`, `ui`.
- Chunking en **tokens** (tokenizer BGE-M3), parent-enfant, `document_id` = hash du chemin relatif et du contenu.
- Qdrant ≥ 1.12 avec vecteurs nommés (`content`, `context`) dès le départ ; OpenSearch gardé pour BM25 tant que le banc d'essai ne prouve pas le contraire.
- Versions (application, schéma d'index, parser, embedding, profil) écrites dans chaque chunk et exposées par `/api/health`.
- Tests unitaires sans GPU ni Docker en CI GitHub Actions ; tests d'intégration et campagnes sur les machines de Damien via `DEV.ps1`.

## 5. Comment on travaille

1. Chaque étape produit des **pull requests** petites, écrites par Claude avec leurs tests, décrites en français (« Avant / Après / Comment vérifier »).
2. GitHub Actions lance les tests unitaires. Rien ne se fusionne en rouge.
3. Quand une étape touche Docker ou le GPU, Damien lance `.\DEV.ps1 -Action test-integration` (ou `smoke`) et colle le résultat dans le fil, ou laisse Claude le faire sur sa machine.
4. Damien fusionne. Claude écrit l'entrée de `docs/JOURNAL.md` et coche l'étape ici.
5. Une étape n'ouvre la suivante que si sa **porte de passage** est franchie.

Fichiers d'organisation : `CLAUDE.md` (règles lues à chaque session), `docs/PLAN.md` (ce fichier), `docs/JOURNAL.md` (une entrée par étape, avec mesures), `docs/DECISIONS.md` (un choix par ligne), `.claude/skills/` (procédures : ouvrir une étape, lancer les tests locaux, clore une étape). La mémoire d'équipe du projet Claude garde les préférences et décisions entre les fils.

## 6. Les étapes

### Étape 0 — Mise en place (1 PR)
Structure, `pyproject.toml` + `uv.lock`, CI, `DEV.ps1`, docs de cadrage, `DECISIONS.md`, `JOURNAL.md`, skills Claude Code, et le socle du cœur : versions, chargement et validation d'un **profil de cas d'usage** (schéma d'entités + boucle en YAML) et d'un **profil matériel**, avec les deux profils `mrae` et `ventes` en squelette.
**Porte :** `uv run pytest` vert dans GitHub Actions et dans PowerShell sur la 3080 Ti.

### Étape 1 — Modèle documentaire et parser PDF (2 à 3 PR)
`DocumentNode`, parser PDF **page par page** (texte natif / Markdown / OCR choisis par page, score de qualité, OCR remis à sa page, boîtes et confiances conservées). Parsers DOCX, texte et transcriptions (réunions). Corpus de test : 4 à 6 PDF non confidentiels + 3 comptes rendus de réunion fictifs.
**Porte :** tests verts sur le corpus ; sur une page A3 dense, texte récupéré ≥ V5.

### Étape 2 — Dossiers, métadonnées typées, chunking (2 PR)
`Dossier`, extraction de métadonnées selon le schéma du profil (règles + LLM à sortie typée), magasin structuré, chunking en tokens, parent-enfant, `document_id` sans collision. Ajout après l'étape 1 : chaque chunk porte aussi le **numéro de page imprimé** quand il existe (les études d'impact impriment deux pages « 14/47 », « 15/47 » par page PDF), pour que les citations parlent le langage du lecteur.
**Porte :** deux fichiers de même nom coexistent ; les champs d'une réunion fictive sont extraits et validés ; aucun chunk ne dépasse la limite du tokenizer.

### Étape 3 — Embedding, index, watcher, multi-architecture (4 PR)
Embedder BGE-M3 (dense `content` + `context`), Qdrant vecteurs nommés, OpenSearch BM25, pipeline, watcher réécrit pour passer les tests portés, images Docker `amd64` + `arm64`, `DEV.ps1 -Action smoke`. Ajout après l'étape 2 : les fichiers du modèle (poids et `tokenizer.json`) sont récupérés une fois à une **révision épinglée**, rangés dans `LUCIOLE_MODELS_DIR`, et `smoke` vérifie leur présence ; l'embedding prend le `embed_text` des morceaux enfants (contexte + passage) et le pipeline se branche sur `scan(..., on_document=...)` pour ne lire chaque fichier qu'une fois.
**Porte (3080 Ti, puis GX10 si disponible) :** ingestion d'un dossier de test, réingestion sans doublon, suppression propre, test de fumée vert.

### Étape 4 — Outils de recherche, génération, verdict, banc d'essai (3 PR)
Outils `retrieve` (hybride + RRF + rerank), `filter`, `aggregate`, `extract`, `synthesize`, `verdict`, gardes déterministes reprises de `query2`, `POST /api/rag/query`. Banc d'essai retrieval (Recall@K, MRR, bonne page) sur le jeu de référence MRAe.
**Porte (GPU) :** pas de régression par rapport à V5 sur bon tome et esquives.

### Étape 5 — Moteur de boucles (2 à 3 PR)
Machine à états pilotée par le YAML du profil, moteur de règles à liste blanche, budget borné, trace complète, `POST /api/agent/query`. Profil `mrae` complet (l'enchaînement de `query2`) et profil `ventes` (chronologie, signaux, comparaison gagnés / perdus).
**Porte :** sur le jeu MRAe, la boucle ne fait jamais moins bien que le RAG seul ; sur 10 dossiers commerciaux fictifs, les analyses citent les bonnes réunions.

### Étape 6 — Interfaces (3 à 4 PR)
Chat par cas d'usage, admin (config, watcher, traces), **éditeur graphique de boucles** (dessiner nœuds, transitions et règles ; enregistrer comme profil ; tester sur une question).
**Porte :** test manuel par Damien ; un profil créé dans l'éditeur change le comportement à la requête suivante.

### Étape 7 — Packaging et hors-ligne, trois machines (2 PR)
Images épinglées par digest, `INSTALL.ps1` / `PREPARE_OFFLINE.ps1` / `MANAGE.ps1` repris avec les profils matériels, manifeste du package.
**Porte :** installation hors-ligne complète sur l'A5000 et le GX10.

## 7. Avancement

- [x] Étape 0
- [x] Étape 1 (2026-10-18, PR #2 à #6)
- [x] Étape 2
- [ ] Étape 3
- [ ] Étape 4
- [ ] Étape 5
- [ ] Étape 6
- [ ] Étape 7
