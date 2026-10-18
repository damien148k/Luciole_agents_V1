"""Prompts d'un cas d'usage : ``configs/profiles/<cas>/prompts.yaml``.

Aucun texte de consigne ne vit dans le code. Un gabarit n'emploie que les
champs que le code remplit ; une accolade littérale s'écrit doublée
(``{{ }}``). Un champ inconnu ou absent est refusé au chargement, pas à la
première question.
"""

from __future__ import annotations

import string
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Champs que chaque gabarit peut employer, et ceux qu'il doit employer.
_FIELDS = {"answer": ({"question", "passages"}, {"question", "passages"})}


def _fields(template: str) -> set[str]:
    try:
        return {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}
    except ValueError as e:
        raise ValueError(f"gabarit illisible ({e}) : doubler les accolades littérales") from e


class PromptSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Rôle et règles du LLM (message système).
    system: str = Field(min_length=1)
    #: Gabarit de la demande : {question} et {passages}.
    answer: str = Field(min_length=1)
    #: Réponse rendue telle quelle quand aucun passage n'est trouvé.
    no_results: str = Field(min_length=1)

    @model_validator(mode="after")
    def _templates(self) -> PromptSet:
        for name, (allowed, required) in _FIELDS.items():
            used = _fields(getattr(self, name))
            unknown, missing = used - allowed, required - used
            if unknown:
                raise ValueError(f"gabarit « {name} » : champ(s) inconnu(s) {sorted(unknown)}")
            if missing:
                raise ValueError(f"gabarit « {name} » : champ(s) absent(s) {sorted(missing)}")
        return self

    def render_answer(self, question: str, passages: str) -> str:
        return self.answer.format(question=question, passages=passages)


def load_prompts(directory: str | Path) -> PromptSet:
    path = Path(directory) / "prompts.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"prompts absents : {path}")
    with path.open(encoding="utf-8") as f:
        return PromptSet.model_validate(yaml.safe_load(f))
