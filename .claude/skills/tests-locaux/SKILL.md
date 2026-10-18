---
name: tests-locaux
description: Lancer et lire les tests de Luciole_agents_V1 sur la machine de Damien (unitaires, intégration, fumée matériel) et interpréter un échec. Utiliser quand une PR touche Docker ou le GPU, ou quand Damien colle une sortie de DEV.ps1.
---

# Tests locaux

Les tests unitaires tournent partout (`uv run pytest`). Ce qui demande Docker ou un GPU porte `@pytest.mark.integration` et ne tourne que sur les machines de Damien.

## Commandes à donner à Damien (PowerShell, racine du dépôt)

```powershell
.\DEV.ps1 -Action install            # une fois, ou après un changement de uv.lock
.\DEV.ps1 -Action test               # unitaires
.\DEV.ps1 -Action smoke              # machine : architecture, GPU, profil matériel suggéré
.\DEV.ps1 -Action test-integration   # Docker + GPU
```

Donner une commande à la fois, dire ce qu'il doit voir, lui demander de coller la sortie complète.

## Lire une sortie

- `uv` absent ou `uv sync` qui échoue : refaire `install` ; si `uv.lock` est en cause, c'est un bug de la PR, pas de sa machine.
- Erreur CUDA ou `torch` : comparer avec le profil matériel (`configs/hardware/`) et l'architecture affichée par `smoke`. Sur le GX10 (aarch64), vérifier que l'image et les roues sont bien en `arm64`.
- Conteneur injoignable : `docker compose ps` puis `docker compose logs <service> --tail 50`.
- Ne jamais proposer de désactiver un test pour passer au vert : corriger la cause, ou noter dans la PR pourquoi le test est faux.

## Après un passage vert

Noter dans `docs/JOURNAL.md` la machine, la date et les chiffres mesurés (durées, nombre de chunks, scores) : c'est ce qui fait foi pour la porte de passage.
