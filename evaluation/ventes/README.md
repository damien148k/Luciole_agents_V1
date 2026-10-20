# Banc « ventes » : 10 dossiers commerciaux fictifs

Dossiers **inventés** (clients, personnes, montants) pour la porte de l'étape 5 : « sur 10 dossiers commerciaux fictifs, les analyses citent les bonnes réunions ». Aucun nom de client réel.

- `dossiers/` : une racine d'ingestion du profil `ventes`, un sous-répertoire par dossier, une transcription par réunion (`reunion-N.transcript.md`). Deux dossiers ont leur fiche dans `dossier.yaml` (numéro de réunion lu dans le nom de fichier), les autres dans le front matter (l'issue de la dernière réunion fait foi).
- `reference.jsonl` : une question par dossier et les réunions qui expliquent son issue (`expected`, champ `part` = numéro de réunion).

| Dossier | Produit | Commercial | Issue | Réunions attendues | Ce qui décide |
|---|---|---|---|---|---|
| bureau-etudes-lambert | Luciole Pro | Camille Martin | gagné | 2, 3 | objection maintenance levée par un contrat de support |
| notaires-girard | Luciole Pro | Julien Bernard | perdu | 2, 3 | décideur jamais vu en démonstration, concurrent intégré moins cher |
| clinique-veterinaire-moreau | Luciole Essentiel | Sofia Garnier | gagné | 3 | pilote d'un mois concluant |
| agence-architecture-fontaine | Luciole Pro | Camille Martin | perdu | 2 | refus du prestataire informatique (pas de local) |
| mairie-saint-aubin | Luciole Pro | Julien Bernard | en cours | 2 | marché public, budget voté en décembre, cahier technique à fournir |
| cabinet-expertise-renaud | Luciole Essentiel | Sofia Garnier | abandonné | 2, 3 | départ de la personne qui portait le projet |
| geometres-blanc | Luciole Pro | Camille Martin | gagné | 2 | prix justifié par le retour sur investissement, paiement étalé |
| industries-perrin | Luciole Pro | Julien Bernard | perdu | 2, 3 | lecture des vieux scans insuffisante |
| laboratoire-chevalier | Luciole Essentiel | Sofia Garnier | perdu | 2, 3 | pas de connecteur vers le logiciel de laboratoire |
| syndic-mercier | Luciole Essentiel | Camille Martin | gagné | 2 | démonstration sur ses propres dossiers |

Mesure (depuis la racine du dépôt) :

```powershell
.\DEV.ps1 -Action index -Root evaluation\ventes\dossiers -UseCase ventes
.\DEV.ps1 -Action bench -UseCase ventes -Reference evaluation\ventes\reference.jsonl -Answer
```
