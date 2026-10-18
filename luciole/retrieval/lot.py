"""Règles du lot de passages, mesurées sur les études d'impact avec V5 (v3.6).

- Un passage de **sommaire** ou de liste de cartes et figures ne porte aucune
  information et fait citer des pages absentes du lot : il est écarté.
- Les **documents secondaires** (résumé non technique, synthèses) sont des
  passages denses que le reranker note mieux que les pages de détail ; sans
  plafond, ils occupent le lot et la réponse reste au niveau du résumé. Au
  plus ``secondary_cap`` passages en viennent ; les places libérées vont aux
  candidats suivants.

Les motifs de nom de fichier et le plafond viennent du profil.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

# Une ligne de sommaire ou de liste de figures : « Carte 12 : … 45 »,
# « 3.2.1 Titre ........ 18 », « Figure 4 – … (page 20) ».
_TOC_LINE = re.compile(
    r"^\s*(?:(?:carte|figure|tableau|photo|planche|annexe|illustration)\s*n?°?\s*\d+\b"
    r"|\d+(?:\.\d+){1,3}\s+\S)"
    r".*?(?:\.{3,}\s*\d{1,3}|\bp(?:age)?\.?\s*\d{1,3}\)?|\s\d{1,3})\s*$",
    re.I,
)


def strip_accents(text: str) -> str:
    """Texte sans accents, apostrophes typographiques ramenées à « ' »."""
    text = (text or "").replace("’", "'").replace("‘", "'")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def is_table_of_contents(text: str, min_lines: int = 5, min_ratio: float = 0.4) -> bool:
    """Vrai si au moins ``min_lines`` lignes de sommaire forment au moins
    ``min_ratio`` des lignes non vides."""
    lines = [line for line in (text or "").splitlines() if line.strip()]
    if len(lines) < min_lines:
        return False
    toc = sum(1 for line in lines if _TOC_LINE.match(line))
    return toc >= min_lines and toc / len(lines) >= min_ratio


def is_secondary(file_name: str, patterns: Sequence[str]) -> bool:
    """Le nom de fichier contient-il un des motifs (sans accents ni casse) ?"""
    name = strip_accents(file_name).lower()
    return any(strip_accents(p).lower() in name for p in patterns if p)
