"""Métadonnées typées : conversion et validation contre le schéma du profil.

Les valeurs arrivent de sources hétérogènes (front matter YAML, nom de
fichier, ``dossier.yaml``, plus tard un LLM à sortie JSON) et sous des formes
variées (« 3 », « 10/02/2026 », « Gagné »). Ce module les ramène à une forme
canonique unique par type, pour que le magasin structuré puisse filtrer,
trier et compter sans surprise :

- ``string`` → texte nettoyé ; ``integer`` → int ; ``number`` → float ;
- ``date`` → ``AAAA-MM-JJ`` (se trie comme une chaîne) ;
- ``boolean`` → True / False (oui, non, vrai, faux…) ;
- ``enum`` → la valeur déclarée, quelle que soit la casse ou l'accentuation.

Une valeur qui ne se convertit pas n'est jamais devinée : elle devient un
problème (``MetadataIssue``) visible dans le rapport d'ingestion.
"""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict

from luciole.core.profile import FieldSpec

IssueLevel = Literal["error", "warning"]

_INT_RE = re.compile(r"^[+-]?\d+$")
_NUMBER_RE = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")
_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d.%m.%Y")
_TRUE = {"true", "vrai", "oui", "yes", "1", "o", "y"}
_FALSE = {"false", "faux", "non", "no", "0", "n"}


class MetadataIssue(BaseModel):
    """Un problème de métadonnée, rattaché à un champ et à sa source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    level: IssueLevel
    field: str
    message: str
    source: str | None = None


def _normalize_token(value: str) -> str:
    """« Gagné », « gagne », « GAGNE » et « en cours » → ``gagne``, ``en_cours``."""
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[\s\-]+", "_", ascii_only.strip().lower())


def coerce_value(spec: FieldSpec, raw: object) -> object | None:
    """Valeur canonique de ``raw`` pour le type de ``spec``.

    Renvoie ``None`` pour une valeur vide (absente) ; lève ``ValueError`` avec
    un message en français si la valeur ne correspond pas au type.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    kind = spec.type
    if kind == "string":
        if isinstance(raw, bool | dict | list):
            raise ValueError(f"texte attendu, reçu {raw!r}")
        return " ".join(str(raw).split())
    if kind == "integer":
        if isinstance(raw, bool):
            raise ValueError(f"entier attendu, reçu {raw!r}")
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float) and raw.is_integer():
            return int(raw)
        if isinstance(raw, str) and _INT_RE.match(raw.strip()):
            return int(raw.strip())
        raise ValueError(f"entier attendu, reçu {raw!r}")
    if kind == "number":
        if isinstance(raw, bool):
            raise ValueError(f"nombre attendu, reçu {raw!r}")
        if isinstance(raw, int | float):
            return float(raw)
        if isinstance(raw, str) and _NUMBER_RE.match(raw.strip()):
            return float(raw.strip().replace(",", "."))
        raise ValueError(f"nombre attendu, reçu {raw!r}")
    if kind == "date":
        if isinstance(raw, dt.datetime):
            return raw.date().isoformat()
        if isinstance(raw, dt.date):
            return raw.isoformat()
        if isinstance(raw, str):
            text = raw.strip()
            for fmt in _DATE_FORMATS:
                try:
                    return dt.datetime.strptime(text, fmt).date().isoformat()
                except ValueError:
                    continue
        raise ValueError(f"date attendue (AAAA-MM-JJ ou JJ/MM/AAAA), reçu {raw!r}")
    if kind == "boolean":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, int) and raw in (0, 1):
            return bool(raw)
        if isinstance(raw, str):
            token = _normalize_token(raw)
            if token in _TRUE:
                return True
            if token in _FALSE:
                return False
        raise ValueError(f"oui ou non attendu, reçu {raw!r}")
    if kind == "enum":
        assert spec.values is not None  # garanti par FieldSpec
        token = _normalize_token(str(raw))
        for allowed in spec.values:
            if _normalize_token(allowed) == token:
                return allowed
        raise ValueError(f"valeur {raw!r} hors de la liste : {', '.join(spec.values)}")
    raise ValueError(f"type de champ inconnu : {kind}")  # pragma: no cover


class ValidatedMetadata(BaseModel):
    """Résultat de la validation : valeurs canoniques, source de chacune,
    champs requis manquants et problèmes rencontrés."""

    model_config = ConfigDict(extra="forbid")

    values: dict[str, object] = {}
    sources: dict[str, str] = {}
    missing: list[str] = []
    issues: list[MetadataIssue] = []

    @property
    def errors(self) -> list[MetadataIssue]:
        return [i for i in self.issues if i.level == "error"]


def merge_sources(
    schema: dict[str, FieldSpec],
    candidates: list[tuple[str, dict[str, object]]],
) -> ValidatedMetadata:
    """Fusionne des sources par ordre de priorité décroissante.

    ``candidates`` est une liste ``(nom_de_source, valeurs_brutes)`` ; pour
    chaque champ du schéma, la première source qui donne une valeur valide
    l'emporte. Une valeur invalide est signalée et la source suivante est
    essayée. Les clés hors schéma sont ignorées (les parsers ajoutent leurs
    propres métadonnées techniques). Un champ requis sans valeur est une
    erreur.
    """
    result = ValidatedMetadata()
    for name, spec in schema.items():
        rejected: list[MetadataIssue] = []
        for source, raw_values in candidates:
            if name not in raw_values:
                continue
            try:
                value = coerce_value(spec, raw_values[name])
            except ValueError as e:
                rejected.append(
                    MetadataIssue(level="error", field=name, message=str(e), source=source)
                )
                continue
            if value is None:
                continue
            result.values[name] = value
            result.sources[name] = source
            break
        if name in result.values:
            # Une source moins prioritaire a comblé le champ : l'invalide n'est
            # plus bloquante, mais reste visible.
            rejected = [i.model_copy(update={"level": "warning"}) for i in rejected]
        elif spec.required:
            result.missing.append(name)
            rejected.append(MetadataIssue(level="error", field=name, message="champ requis absent"))
        result.issues.extend(rejected)
    return result
