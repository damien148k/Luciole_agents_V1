"""Prompts d'un cas d'usage : ``configs/profiles/<cas>/prompts.yaml``.

Aucun texte de consigne ne vit dans le code. Un gabarit n'emploie que les
champs que le code remplit ; une accolade littérale s'écrit doublée
(``{{ }}``). Un champ inconnu ou absent est refusé au chargement, pas à la
première question.
"""

from __future__ import annotations

import string
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

#: Champs que chaque gabarit peut employer, et ceux qu'il doit employer.
_FIELDS = {"answer": ({"question", "passages", "framing"}, {"question", "passages"})}


def _fields(template: str) -> set[str]:
    try:
        return {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}
    except ValueError as e:
        raise ValueError(f"gabarit illisible ({e}) : doubler les accolades littérales") from e


class ExtractPrompt(BaseModel):
    """Une extraction à sortie typée (outil ``extract`` des boucles).

    ``fields`` décrit le JSON attendu : ``string``, ``list`` (liste de textes)
    ou une liste de valeurs permises (énumération). Le gabarit ``prompt``
    reçoit ``{question}``."""

    model_config = ConfigDict(extra="forbid")

    system: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    fields: dict[str, Literal["string", "list"] | list[str]] = Field(min_length=1)

    @model_validator(mode="after")
    def _template(self) -> ExtractPrompt:
        used = _fields(self.prompt)
        if used != {"question"}:
            raise ValueError(f"gabarit d'extraction : seul {{question}} est permis, pas {used}")
        for name, kind in self.fields.items():
            if isinstance(kind, list) and not kind:
                raise ValueError(f"champ « {name} » : énumération vide")
        return self

    def model(self) -> type[BaseModel]:
        """Le modèle pydantic du JSON attendu (champs tous obligatoires)."""
        spec: dict = {}
        for name, kind in self.fields.items():
            if kind == "string":
                spec[name] = (str, ...)
            elif kind == "list":
                spec[name] = (list[str], ...)
            else:
                spec[name] = (Literal[tuple(kind)], ...)
        return create_model("Extraction", __config__=ConfigDict(extra="forbid"), **spec)


class PromptSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Rôle et règles du LLM (message système).
    system: str = Field(min_length=1)
    #: Gabarit de la demande : {question} et {passages}.
    answer: str = Field(min_length=1)
    #: Réponse rendue telle quelle quand aucun passage n'est trouvé.
    no_results: str = Field(min_length=1)
    #: Extractions typées des boucles (outil ``extract``), par nom de schéma.
    extract: dict[str, ExtractPrompt] = Field(default_factory=dict)
    #: Gabarits de questions de recherche (outil ``compose``) ; leurs champs
    #: sont ceux des extractions, plus ``aspect`` et ``motif``.
    questions: dict[str, str] = Field(default_factory=dict)
    #: Cadrage ajouté à la rédaction (``{framing}`` du gabarit ``answer``) ;
    #: ses champs sont ceux des extractions.
    framing: str = ""

    @model_validator(mode="after")
    def _templates(self) -> PromptSet:
        for name, (allowed, required) in _FIELDS.items():
            used = _fields(getattr(self, name))
            unknown, missing = used - allowed, required - used
            if unknown:
                raise ValueError(f"gabarit « {name} » : champ(s) inconnu(s) {sorted(unknown)}")
            if missing:
                raise ValueError(f"gabarit « {name} » : champ(s) absent(s) {sorted(missing)}")
        extracted = {f for e in self.extract.values() for f in e.fields}
        for name, template in self.questions.items():
            unknown = _fields(template) - extracted - {"aspect", "motif"}
            if unknown:
                raise ValueError(f"question « {name} » : champ(s) inconnu(s) {sorted(unknown)}")
        unknown = _fields(self.framing) - extracted
        if unknown:
            raise ValueError(f"cadrage : champ(s) inconnu(s) {sorted(unknown)}")
        return self

    def render_answer(self, question: str, passages: str, framing: str = "") -> str:
        if "{framing}" not in self.answer:
            return self.answer.format(question=question, passages=passages)
        return self.answer.format(question=question, passages=passages, framing=framing)

    def render_framing(self, values: dict) -> str:
        """Le cadrage rempli avec une extraction ; les listes sont jointes."""
        if not self.framing:
            return ""
        return self.framing.format(**{k: as_text(v) for k, v in values.items()}).strip()


def as_text(value) -> str:
    """Une valeur d'extraction dans un gabarit : une liste devient « a ; b »."""
    if isinstance(value, list):
        return " ; ".join(value) if value else "aucun"
    return str(value)


def load_prompts(directory: str | Path) -> PromptSet:
    path = Path(directory) / "prompts.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"prompts absents : {path}")
    with path.open(encoding="utf-8") as f:
        return PromptSet.model_validate(yaml.safe_load(f))
