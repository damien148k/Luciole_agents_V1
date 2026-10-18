# Luciole_agents_V1

Plateforme souveraine et hors ligne de **boucles agentiques par cas d'usage**, dont le RAG est un des outils. Refonte complète de Luciole Prime V5 (`damien148k/luciole-prime-v5`) : on reprend ses idées d'architecture et ses leçons mesurées, pas son code. Cas d'usage connus : réponses aux avis MRAe (études d'impact), assistant des commerciaux (suites de réunions client, dossiers gagnés ou perdus).

## À lire avant de toucher au code

- Le propriétaire, Damien, n'est pas développeur de métier. Chaque PR explique en français simple ce qu'elle change et comment la vérifier, avec la commande PowerShell à coller.
- On avance **une étape à la fois** selon `docs/PLAN.md`. Une étape n'est ouverte que si la porte de passage de la précédente est notée dans `docs/JOURNAL.md`.
- Tout choix technique va dans `docs/DECISIONS.md`. On ne rediscute pas un choix noté sans nouvelle mesure.
- Les documents de `docs/cadrage/` font foi : `architecture.md`, `rag-ingestion-spec.md`, `agent-harness-architecture.md`, `postmortem-belacom.md`, `analyse-v5.md`.

## Règles non négociables

1. **Hors ligne en production** : aucun appel à une API cloud dans le chemin de production. Modèles chargés depuis le cache local.
2. **Rien de figé en dur** : schéma d'entités, chunking, recherche, seuils, règles et boucles viennent du profil de cas d'usage (`configs/profiles/<cas>/`) et du profil matériel (`configs/hardware/<machine>.yaml`).
3. **Le code tient la boucle, pas le LLM** : une boucle est une machine à états en code, décrite en YAML ; le LLM n'exécute que des tâches bornées à sortie typée (JSON validé). Jamais de boucle ReAct libre (voir le post-mortem Belacom).
4. **Traçabilité** : hash du document, dossier, page, section, parent, méthode d'extraction et versions dans chaque chunk ; trace complète par requête.
5. **Pas de test sauté, désactivé ou affaibli pour passer au vert.**
6. **Dépendances verrouillées** : `uv.lock` fait foi ; images Docker épinglées par version ou digest, construites pour `amd64` et `arm64`. Aucun `latest`.
7. **Trois machines** (3080 Ti 12 Go, A5000 24 Go, GX10 ARM64 128 Go) : tout réglage qui dépend du matériel passe par un profil matériel, jamais par une constante.

## Commandes

```powershell
.\DEV.ps1 -Action install          # créer l'environnement (uv)
.\DEV.ps1 -Action test             # tests unitaires (sans GPU ni Docker)
.\DEV.ps1 -Action lint             # ruff
.\DEV.ps1 -Action test-integration # tests Docker + GPU (machines de Damien)
.\DEV.ps1 -Action models           # récupérer une fois les modèles (révision épinglée)
.\DEV.ps1 -Action index -Root <dossier> -UseCase ventes  # indexer une racine de dossiers
.\DEV.ps1 -Action watch -Root <dossier> -UseCase ventes  # garder l'index à jour
.\DEV.ps1 -Action search -UseCase ventes -Query "question"  # chercher et citer
.\DEV.ps1 -Action ask -UseCase ventes -Query "question"     # réponse sourcée (Ollama), verdict, trace
.\DEV.ps1 -Action api                                       # API http://127.0.0.1:8000
.\DEV.ps1 -Action smoke            # test de fumée dans l'image Docker (GPU, modèles, services)
```

Équivalents : `uv run pytest`, `uv run --extra embed pytest -m integration`, `uv run ruff check .`.

## Structure

```
luciole/
  core/        # DocumentNode, Dossier, profils (cas d'usage, matériel), versions
  ingestion/   # parsers (PDF page par page, DOCX, texte, transcriptions), OCR, chunking
  store/       # magasin structuré des métadonnées (filtrer, compter, grouper)
  index/       # embedder BGE-M3, Qdrant (vecteurs nommés), OpenSearch BM25, pipeline, watcher
  retrieval/   # hybride, RRF, reranker, gardes déterministes
  generation/  # LLM local (Ollama), prompts, sorties typées, verdict
  agents/      # moteur de boucles : machine à états, outils, règles, traces
  api/         # FastAPI : /api/rag/query, /api/agent/query, /api/health
  ui/          # chat, admin, éditeur de boucles (HTML/JS dans des fichiers séparés)
configs/
  profiles/<cas>/   # profile.yaml (entités, ingestion, recherche), loop.yaml, prompts.yaml, evaluation/
  hardware/         # rtx3080ti.yaml, a5000.yaml, gx10.yaml
tests/              # unitaires par défaut ; marqueur `integration` pour Docker/GPU
docs/               # PLAN, JOURNAL, DECISIONS, cadrage/
evaluation/         # bancs d'essai retrieval et RAGAS
```

## Conventions

- Code, identifiants et commits en anglais ; commentaires, docstrings, docs et messages utilisateur en français.
- Un module = une responsabilité ; les clients externes (Qdrant, OpenSearch, Ollama, modèles) sont injectables pour que les tests unitaires tournent sans eux.
- Les tests unitaires ne téléchargent rien et durent quelques secondes. Ce qui a besoin de Docker ou d'un GPU porte `@pytest.mark.integration`.
- Une PR = un sujet. Titre en anglais, description en français : « Avant / Après / Comment vérifier ».
- **Une seule PR ouverte à la fois, toujours sur `main`.** Pas de PR empilées : fusionnées par le bouton vert, elles atterrissent dans la branche intermédiaire, pas dans `main` (vécu à l'étape 1).
- Jamais de secret, de chemin personnel, de nom de client réel ni de données clients dans le code ou les fixtures. Les dossiers commerciaux de test sont fictifs.

## Leçons de V5 à ne pas oublier

- `document_id` doit être unique même pour deux fichiers de même nom (hash du chemin relatif et du contenu).
- Le texte OCR va à la place de sa page, pas en fin de document.
- Les tailles de chunk se mesurent en tokens du modèle d'embedding, pas en caractères.
- Les poids sparse et ColBERT de BGE-M3 ne servent que si le code les charge ; ne pas les embarquer sans usage.
- pymupdf4llm peut perdre silencieusement le texte des pages A3 denses : garder une comparaison de rendement par page.
- La détection « PDF vectoriel complexe » et l'OCR étendu doivent se décider page par page, pas sur la moyenne du document.

## Leçons des études d'impact réelles (étape 1)

- Les études sont des PDF portrait tournés de 90°, imprimés « deux pages côte à côte » : toute coordonnée de bloc passe par `page.rotation_matrix`, et les numéros imprimés (« 14/47 ») ne sont pas les numéros PDF.
- Une page sans texte, sans image et sans tracé est vide, pas à OCR. Un tableau tracé fait jusqu'à ~2000 tracés, une carte 7000 et plus : ne pas baisser le seuil de 5000 sans nouvelle mesure.
- Les seuils du parser se règlent sur des documents réels publics, jamais sur le seul corpus synthétique ; les documents réels ne rentrent pas dans le dépôt.

## Leçons de l'étape 2

- Les tailles de morceaux se vérifient avec le tokenizer réel, mais les tests unitaires utilisent un compteur factice : le conteneur cloud de Claude ne joint pas huggingface.co, seule la machine de Damien peut lancer les tests `integration` avec BGE-M3.
- Une valeur de métadonnée invalide est signalée, jamais corrigée ni devinée ; un dépassement de taille lève une erreur, jamais une troncature silencieuse.

## Leçons de l'étape 3

- Le GPU dans Docker demande les capacités `compute` (CUDA) et `utility` (nvidia-smi) : nvidia-smi qui marche dans le conteneur ne prouve pas que PyTorch voit le GPU.
- Le cache de uv se monte pendant la construction (`--mount=type=cache`), sinon chaque bibliothèque CUDA est copiée deux fois dans l'image.
