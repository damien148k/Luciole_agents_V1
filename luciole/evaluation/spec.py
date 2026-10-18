"""Réglages du banc d'essai d'un cas d'usage : ``configs/profiles/<cas>/evaluation/bench.yaml``.

Rien n'est figé dans le code : la forme des réponses de référence (marque
d'une remarque, renvoi à un tome, numéro de page), le champ de document qui
porte le tome et les seuils de mesure viennent de ce fichier.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import Field, field_validator

from luciole.core.profile import StrictModel


def _regex(value: str) -> str:
    try:
        re.compile(value)
    except re.error as e:
        raise ValueError(f"expression régulière illisible « {value} » : {e}") from e
    return value


class ReferenceSpec(StrictModel):
    """Lecture d'une réponse de référence (DOCX) en questions et renvois attendus."""

    remark_pattern: str = Field(
        description="Début d'une remarque, avec un groupe pour son numéro (cellule de tableau)"
    )
    part_pattern: str = Field(description="Renvoi à une pièce, avec un groupe pour son numéro")
    page_pattern: str = Field(
        description="Page ou plage de pages, groupes : première page, dernière page (optionnel)"
    )
    page_more_pattern: str = Field(
        default=r"^\s*(?:\([^)]*\)\s*)?et\s*(\d{1,4})\b",
        description="Page de plus juste après une page (« p. 331 et 393 »), un groupe",
    )
    max_range: int = Field(default=30, ge=0, description="Plage plus longue : sa première page")

    _check = field_validator("remark_pattern", "part_pattern", "page_pattern", "page_more_pattern")(
        _regex
    )


class MetricsSpec(StrictModel):
    part_field: str = Field(description="Champ de document qui porte le numéro de pièce (tome)")
    k: list[int] = Field(default=[1, 3, 8], min_length=1)
    page_tolerance: int = Field(default=1, ge=0, description="Écart permis entre pages")

    @field_validator("k")
    @classmethod
    def _positive(cls, value: list[int]) -> list[int]:
        if any(k < 1 for k in value):
            raise ValueError("k doit être au moins 1")
        return sorted(set(value))


class BenchSpec(StrictModel):
    reference: ReferenceSpec
    metrics: MetricsSpec
    #: Résultats connus de la version précédente, recopiés tels quels dans le rapport.
    baseline: dict[str, str] = Field(default_factory=dict)


def load_bench_spec(profile_dir: str | Path) -> BenchSpec:
    path = Path(profile_dir) / "evaluation" / "bench.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"réglages du banc d'essai absents : {path}")
    with path.open(encoding="utf-8") as f:
        return BenchSpec.model_validate(yaml.safe_load(f))
