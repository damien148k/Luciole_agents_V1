---
name: clore-etape
description: Clore une étape du plan Luciole_agents_V1 une fois sa porte de passage franchie (journal, plan, décisions, mémoire). Utiliser quand la dernière PR d'une étape est fusionnée et vérifiée.
---

# Clore une étape

1. Vérifier que la porte de passage de `docs/PLAN.md` est réellement franchie, avec une preuve : sortie de CI verte, et pour les étapes Docker/GPU la sortie collée par Damien (ou obtenue sur sa machine). Pas de porte franchie « de mémoire ».
2. Compléter l'entrée de l'étape dans `docs/JOURNAL.md` : ce qui a été fait (PR fusionnées), ce qui a été mesuré (chiffres, machine, date), ce qui a été décidé, ce qui reste ou a été reporté.
3. Ajouter dans `docs/DECISIONS.md` chaque choix pris pendant l'étape qui ne s'y trouve pas encore (une ligne : date, décision, raison).
4. Cocher l'étape dans « Avancement » de `docs/PLAN.md`. Si l'étape a appris quelque chose qui change les suivantes, modifier leur description et le dire dans le journal.
5. Si une leçon doit survivre aux sessions (préférence de Damien, piège technique), l'ajouter à `CLAUDE.md` en une ligne et à la mémoire d'équipe du projet.
6. Ouvrir ces mises à jour dans une petite PR « docs: close step N » et dire à Damien en une phrase ce qui a été franchi et ce que la prochaine étape attend de lui.
