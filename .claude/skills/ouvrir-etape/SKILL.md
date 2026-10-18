---
name: ouvrir-etape
description: Ouvrir une étape du plan Luciole_agents_V1 (vérifier la porte précédente, créer la branche, cadrer la PR). Utiliser quand Damien dit « on passe à l'étape N » ou « ouvre l'étape N ».
---

# Ouvrir une étape

1. Lire `docs/PLAN.md` et `docs/JOURNAL.md`. Vérifier que la porte de passage de l'étape précédente est notée comme franchie dans le journal. Si elle ne l'est pas, le dire à Damien et s'arrêter : on n'ouvre pas une étape sur une porte non franchie.
2. Relire la section de l'étape dans `docs/PLAN.md` et les leçons de `CLAUDE.md`. Si l'étape touche au PDF, relire `docs/cadrage/analyse-v5.md` ; si elle touche à la boucle, relire `docs/cadrage/agent-harness-architecture.md`.
3. Découper l'étape en PR d'un seul sujet (le plan donne un ordre de grandeur). Pour chaque PR : ce qu'elle contient, ses tests, et comment Damien la vérifie (commande `DEV.ps1`).
4. Créer la branche `etape-N-<sujet>` depuis `main` à jour, ajouter dans `docs/JOURNAL.md` l'entrée « Étape N (ouverte le JJ-MM-AAAA) » avec la porte de passage recopiée.
5. Écrire les tests avant ou avec le code. Un test ne se saute ni ne se désactive.
6. Ouvrir la PR en brouillon avec la description « Avant / Après / Comment vérifier » en français, puis la suivre jusqu'au vert.
