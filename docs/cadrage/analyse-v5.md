# Vérification de la critique Copilot de Luciole Prime V5

Code lu : `damien148k/luciole-prime-v5`, branche `main`, commit `91d9d34` (30 septembre 2026). Les références `fichier:ligne` renvoient à ce commit.

Verdicts : **Confirmé** (le code montre ce que dit la critique), **Nuancé** (vrai en partie, ou formulé de façon trompeuse), **Infirmé** (le code dit le contraire).

---

## 1. Le gitingest et le diagramme sont-ils utiles pour Claude ?

**Le gitingest** (1,09 Mo, environ 300 000 tokens) est un instantané fidèle, mais trop gros pour une conversation classique. Copilot l'avoue d'ailleurs lui-même en section 15 : il n'a pas pu ouvrir le fichier en entier et a travaillé sur des extraits. Une bonne partie de sa critique repose donc sur des suppositions, ce qui explique les formulations prudentes (« semble », « à vérifier »).

Pour Claude, la meilleure solution est de lui donner accès au dépôt directement, comme je l'ai fait ici en le clonant. Il lit alors seulement les fichiers utiles, à leur version du jour. Si tu veux un format à coller dans un chat sans accès au dépôt, trois choses aident plus que le gitingest complet :

1. **Un `CLAUDE.md` à la racine** (il n'en existe aucun aujourd'hui) : une à deux pages qui donnent l'architecture réelle, le chemin d'un document de `data/` jusqu'à Qdrant et OpenSearch, les commandes de test, les conventions (français, sans accents dans les commentaires), et ce qui vient de V4 par rapport à V5.
2. **Un README à jour** : il dit aujourd'hui « Squelette initial… aucune logique métier » (`README.md:89`), ce qui induit en erreur n'importe quel lecteur, humain ou IA.
3. **Un gitingest filtré** si besoin : sans `src/ui/` (plus de 10 000 lignes d'interface sur 22 000), sans les polices ni les images. Il passe alors à environ un tiers de la taille actuelle.

**Le diagramme Mermaid** (généré par gitdiagram) est globalement juste. Il contient une petite erreur : c'est `retriever.py` qui appelle le reranker (`src/rag/retriever.py:89`), pas `hybrid.py`, et c'est `hybrid.py` qui appelle les recherches dense et BM25 (`src/rag/hybrid.py:81-82`), pas le retriever. Il ne montre pas non plus la branche OCR, ni le fait que l'Excel passe par le parser et pas directement par le pipeline.

---

## 2. Points forts (section 4)

| Point | Verdict | Preuve |
|---|---|---|
| 4.1 Séparation des responsabilités | **Confirmé** | `src/watcher/`, `src/ingestion/`, `src/rag/` et `src/ui/` sont bien séparés. Le pipeline accepte des doublures pour Qdrant, OpenSearch et l'embedder (`src/ingestion/pipeline.py:94-103`), ce qui permet les tests sans GPU. |
| 4.2 Recherche hybride + RRF + reranking | **Confirmé** | RRF pondéré avec k=60 (`src/rag/hybrid.py:111-125`), puis cross-encoder `bge-reranker-v2-m3` (`src/rag/reranker.py:44`). |
| 4.3 Traçabilité des pages | **Confirmé** | Table `page_spans` construite par le parser (`parsers.py:205-258`), `page_start`/`page_end` dans chaque chunk (`chunker.py:291-297`), étiquette « [Source: X, page N] » dans le contexte du LLM (`retriever.py:113-123`). Les tests `test_pages.py` et `test_projection_pages.py` existent. |
| 4.4 Test de rendement PDF | **Confirmé**, avec sa limite | `tests/ingestion/test_rendement_pdf.py` existe, mais il ne vérifie que la longueur du texte (voir 5.3). |
| 4.5 Résilience du watcher | **Confirmé** | `tests/watcher/test_crash_recovery.py`, `test_queue_concurrency.py`, `test_reconciler_integration.py`, etc. Le worker supprime les anciens chunks avant de réindexer (`src/watcher/worker.py:304`), ce qui évite les chunks orphelins. |
| 4.6 Configuration par instance | **Confirmé** | Chunking par type de document dans `configs/settings.yaml.example:58-64`, `rerank_top_n` lu dans les réglages à chaque requête (`retriever.py:51-53`). |

---

## 3. Incohérences et limites (section 5)

### 5.1 Décalage V4 / V5 et README « squelette » : **Confirmé, et c'est pire que ce que dit la critique**

- Le README annonce « Squelette initial du dépôt. Aucune logique métier n'est encore implémentée » (`README.md:89`), alors que le dépôt contient environ 22 000 lignes de Python fonctionnel.
- Le même README parle de « Modèle V4 » (`README.md:23`).
- L'API se présente comme `FastAPI(title="Luciole (cadrage repris dans Luciole_agents_V1) — RAG", version="4.0.0")` (`src/rag/api.py:25`).
- `requirements.txt:1`, `INSTALL.ps1:3`, `docker-compose.yml:1` portent tous l'en-tête « Luciole (cadrage repris dans Luciole_agents_V1) », et l'image Docker s'appelle `luciole-v4:latest` (`docker-compose.yml:19`, `INSTALL.ps1:311-327`).
- Il y a même trois générations visibles : les commentaires des modules parlent de « V3 » (`embedder.py:4`, `ocr.py:5`, `chunker.py:3`).
- La spec du dépôt dit « Ne réutilise pas le code `query2` existant » (`docs/rag-ingestion-spec.md:3`), alors que le retriever l'importe (`src/rag/retriever.py:26`) et que le service RAG passe par lui.
- Aucune version de schéma d'index n'est enregistrée : `_init_qdrant` réutilise une collection existante sans vérifier ni sa dimension, ni le modèle d'embedding qui l'a produite (`pipeline.py:165-169`). Le health check ne publie aucune version (`api.py:79-85`).

La recommandation de versionner le schéma, le parser et l'embedding est justifiée.

### 5.2 Modèle documentaire trop textuel : **Confirmé**

Tous les parsers renvoient `{"content": <une chaîne>, "metadata": {...}}` (le contrat de `BaseParser.parse`, `parsers.py:73`). Pour le PDF, le contenu est une seule chaîne Markdown, accompagnée seulement de métadonnées globales et de `page_spans` (`parsers.py:400-421`). On ne garde ni les blocs, ni les coordonnées, ni le type de bloc, ni le chemin de section, ni la méthode d'extraction page par page. Seul `extraction_method` est conservé, pour tout le document.

Le chunker doit ensuite redécouvrir la structure : il redétecte les tableaux Markdown par expression régulière pour réinjecter leurs en-têtes (`chunker.py:549-630`), ce qui illustre exactement le problème décrit.

La propre spec du dépôt demandait déjà `section/titre` et `parent_chunk_id` dans les métadonnées de chunk (`docs/rag-ingestion-spec.md:45`). Ces champs n'existent pas dans le code. La critique rejoint donc ta propre spec.

### 5.3 Contrôle PDF fondé sur la longueur du texte : **Confirmé**

Le garde-fou compare seulement des longueurs : si le Markdown d'une page fait moins de 50 % du texte brut et que le texte brut dépasse 200 caractères, la page est réextraite en texte simple (`parsers.py:333`). Rien ne vérifie l'ordre de lecture, les colonnes ou les tableaux. Le test associé ne mesure lui aussi que des longueurs.

Une précision : le garde-fou corrige une perte réelle et documentée (environ 35 % du texte récupéré sur des pages A3 paysage, voir `parsers.py:316-324`). Il est utile. Il ne suffit simplement pas comme score de qualité.

### 5.4 Routage PDF global plutôt que page par page : **Nuancé**

- **Vrai pour les PDF vectoriels complexes.** La détection calcule une moyenne de dessins par page sur tout le document (`parsers.py:146-179`, seuil à la ligne 171). Si le document est jugé complexe, il passe en entier en extraction simple, sans Markdown ni tableaux (`parsers.py:368`). Une seule page de plan peut donc dégrader tout un tome, et inversement une page de plan peut passer inaperçue dans un document long.
- **Vrai pour l'« OCR étendu ».** Si le texte total est trop court, l'OCR est lancé sur toutes les pages (`parsers.py:385`).
- **Faux pour le reste.** L'extraction Markdown passe par lots de 10 pages, avec repli page par page (`parsers.py:260-312`). La détection des pages scannées et le garde-fou de rendement fonctionnent aussi page par page (`parsers.py:438-451`, `314-340`). Une partie du routage page par page existe donc déjà.

### 5.5 OCR monolithique : **Confirmé, et il y a plus urgent que ce que dit la critique**

- **L'OCR est désactivé par défaut.** On trouve `enable_ocr: false` dans `configs/settings.yaml.example:67` et dans les valeurs par défaut de `src/config/settings.py:52`. `INSTALL.ps1` recopie cet exemple (`INSTALL.ps1:250-254`). En production, les pages scannées ne sont donc pas lues du tout, sauf si l'OCR a été activé à la main.
- Un seul moteur est utilisé, EasyOCR (`ocr.py:17`, `ocr.py:86`). Aucun moteur n'est choisi selon le type de page.
- Les boîtes et les scores de confiance sont lus, puis jetés : seul le texte est gardé (`ocr.py:112-116`). La langue détectée n'est pas conservée.
- **Défaut non relevé par Copilot.** Le texte OCR est ajouté à la fin du document, et non à la place de la page concernée (`parsers.py:491-510`). Cela casse l'ordre de lecture : un chunk peut mélanger la fin du document et une page OCR du début. Le texte natif de la page reste aussi en place, donc il peut y avoir des doublons. Le marqueur `--- OCR Page N ---` est injecté dans le texte (`parsers.py:494`), alors que le code retire ailleurs ce type de marqueur précisément pour ne pas polluer les embeddings (`parsers.py:189-191`).

### 5.6 BGE-M3 sous-exploité : **Confirmé**

- BGE-M3 est chargé avec `SentenceTransformer` (`embedder.py:117`), qui ne produit que le vecteur dense. `FlagEmbedding` n'est utilisé nulle part dans le dépôt, et aucun appel ne sort de vecteur sparse ni ColBERT.
- La collection Qdrant n'a qu'un seul vecteur sans nom (`pipeline.py:172`, `pipeline.py:366`).
- La seule trace du sparse et de ColBERT se trouve dans `setup_bge_model.py:48-49`, qui télécharge `sparse_linear.pt` et `colbert_linear.pt`. Ces fichiers ne sont jamais chargés : ils occupent de la place dans le package hors ligne sans servir à rien.
- **Point bloquant non relevé par Copilot.** Le serveur Qdrant est épinglé en `v1.7.4` (`docker-compose.yml:55`). Cette version gère les vecteurs nommés et les vecteurs sparse, mais pas les multivecteurs nécessaires au late interaction de type ColBERT, qui arrivent avec Qdrant 1.10. Les étapes 6 et 7 du plan Copilot supposent donc une montée de version de Qdrant. Le client, lui, est en `qdrant-client>=1.9.0` (`requirements.txt:36`), donc déjà plus récent que le serveur.

### 5.7 Un vecteur unique pour plusieurs intentions : **Confirmé**

L'embedding est calculé sur `text_with_context`, c'est-à-dire « [Fichier | Chemin | Type | Titre | Auteur] » suivi du passage (`embedder.py:177`, `chunker.py:164-186`, `chunker.py:284`). Il n'existe qu'un seul vecteur, qui mélange contexte et contenu. À noter : le contexte ajouté contient le nom et le chemin du fichier, mais pas le chemin de section, puisque celui-ci n'existe pas (voir 5.2).

### 5.8 Le chunk comme unité universelle : **Confirmé**

Le même `text` sert à BM25 (`bm25_search.py:57`), au reranking (`reranker.py:74`), au contexte du LLM (`retriever.py:123`) et à la citation. Il n'y a ni parent, ni voisinage, ni `parent_chunk_id`, alors que ta spec le prévoyait.

### 5.9 Chunking en caractères plutôt qu'en tokens : **Confirmé**

Toutes les tailles sont des longueurs Python en caractères : `len(current_chunk) + len(sentence) > cs` (`chunker.py:486`), et de même pour les paragraphes et `_split_oversized`. Le budget de l'embedder est lui aussi exprimé en caractères (`embedder.py:85`). Aucun tokenizer n'est appelé.

Ta spec parle pourtant de « 512–1024 tokens » pour la prose (`docs/rag-ingestion-spec.md:34`). Le réglage actuel, 1024 caractères pour les PDF (`settings.yaml.example:59`), représente environ 250 à 300 tokens en français, soit nettement moins que ce que prévoit la spec. Cet écart mérite d'être tranché par un test, pas seulement documenté.

### 5.10 Dépendances non verrouillées : **Confirmé**

- `ollama/ollama:latest` (`docker-compose.yml:26`) et l'image applicative `luciole-v4:latest`. Qdrant (`v1.7.4`), OpenSearch (`2.11.0`) et la base CUDA (`Dockerfile:4`) sont, eux, correctement épinglés.
- Dans `requirements.txt`, presque tout est en `>=` (`sentence-transformers>=3.0.0`, `pymupdf4llm>=1.28.0`, `easyocr>=1.7.1`…). Il n'y a ni lock ni hashes. Seuls `torch`, `extract-msg` et la chaîne RAGAS/LangChain sont épinglés.
- La révision du modèle n'est pas fixée : `_get_local_model_path` prend le premier snapshot trouvé (`embedder.py:61-64`).
- Le dépôt a déjà été victime de ce problème : le commentaire de `requirements.txt:4-15` raconte la casse du 7 septembre 2026, quand pip a remplacé torch. C'est la meilleure preuve que ce point est réel.

### Autre problème non relevé par Copilot : collision de noms de fichiers

`document_id` est le simple nom du fichier, sans le chemin (`chunker.py:137`, `worker.py:15`). Deux `Tome_4.pdf` placés dans deux sous-dossiers différents reçoivent les mêmes `chunk_id` (`chunker.py:301`) : ils s'écrasent l'un l'autre dans Qdrant et OpenSearch, et supprimer l'un supprime aussi l'autre (`worker.py:304`). Pour des dossiers MRAe où chaque projet a son « Tome 4 », le risque est réel.

---

## 4. Avis sur l'ordre de priorités (section 13)

L'ordre proposé par Copilot est : 1. versions et reproductibilité, 2. DocumentNode, 3. routage page par page, 4. parent-enfant, 5. multi-dense, 6. sparse, 7. late interaction.

**Les étapes justifiées par le code :**

- **1. Versions et reproductibilité** : justifié (voir 5.1 et 5.10). C'est peu coûteux et cela évite des diagnostics faux.
- **2. DocumentNode** : justifié (voir 5.2), et ta propre spec le demandait déjà en partie.
- **3. Routage page par page** : justifié pour la détection des PDF vectoriels et l'OCR étendu (voir 5.4). Le reste fonctionne déjà page par page, donc le chantier est plus petit qu'annoncé.
- **4. Parent-enfant** : justifié (voir 5.8).

**Mon ordre, ajusté :**

0. **Correctifs immédiats, avant tout chantier** : rendre `document_id` unique (chemin relatif ou hash), réinsérer le texte OCR à sa page plutôt qu'en fin de document, et décider explicitement si l'OCR doit être activé par défaut. Ce sont des défauts de justesse, pas des améliorations, et ils fausseraient toute campagne d'ablation.
1. Versions, lock des dépendances, `README` et `CLAUDE.md` à jour.
2. **Banc d'évaluation du retrieval** (Recall@K, bonne page, MRR). Copilot le place en troisième « priorité » dans sa synthèse (section 2) mais l'oublie dans l'ordre de la section 13. Sans ce banc, on ne peut pas savoir si les étapes suivantes apportent quelque chose. Le jeu de référence et `campagne_reference.py` existent déjà, mais ils sont orientés RAGAS, donc vers la génération.
3. DocumentNode, puis routage page par page (en un seul chantier, sur le PDF uniquement).
4. Chunking aligné sur les tokens et parent-enfant (en même temps, car les deux touchent le chunker).
5. Multi-dense nommé (`content` / `context`), possible avec Qdrant 1.7.4.
6. Sparse BGE-M3 : il demande de passer de `SentenceTransformer` à `FlagEmbedding` ou à un équivalent. C'est possible sur Qdrant 1.7.4.
7. Late interaction : il demande d'abord de monter Qdrant au moins en 1.10, et d'avoir prouvé que les étapes 5 et 6 ne suffisent pas.

En résumé, la critique de Copilot est solide sur le fond : je n'ai trouvé aucun point infirmé, et seul le 5.4 est surestimé. En revanche, elle manque trois défauts concrets (collision de `document_id`, OCR placé en fin de document et désactivé par défaut, Qdrant trop ancien pour ColBERT) qui passent avant ses chantiers d'architecture.
