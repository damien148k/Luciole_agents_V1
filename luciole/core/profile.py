"""Profil de cas d'usage : l'objet central de Luciole_agents_V1.

Un cas d'usage (réponses aux avis MRAe, assistant des commerciaux…) est un
dossier ``configs/profiles/<cas>/`` qui contient :

- ``profile.yaml`` : identité du cas, schéma des entités (quels champs
  décrivent un document et un dossier), réglages d'ingestion et de recherche ;
- ``loop.yaml`` : la boucle agentique, machine à états décrite en YAML.

Ce module charge et **valide** ces fichiers. Il ne les interprète pas encore :
le moteur de boucles arrive à l'étape 5. Valider dès l'étape 0 fixe le
contrat que l'éditeur graphique (étape 6) devra respecter.

Règle héritée du post-mortem Belacom : tout ce qui est déclaré ici est
réellement exécuté par le code, ou refusé à la validation. Aucune clé
« documentaire » qui ne ferait rien.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Types de champs acceptés dans un schéma d'entités.
FieldType = Literal["string", "integer", "number", "date", "boolean", "enum"]

# Outils que le moteur de boucles saura exécuter. Un nœud qui en nomme un
# autre est refusé à la validation (pas d'outil « à venir » toléré en YAML).
ToolName = Literal[
    "retrieve",  # recherche hybride + rerank
    "filter",  # magasin structuré : sélectionner des dossiers/documents
    "aggregate",  # magasin structuré : compter, grouper
    "extract",  # LLM à sortie JSON validée contre un schéma
    "compare",  # mise en regard déterministe de deux ensembles
    "synthesize",  # LLM : rédaction à partir d'extraits cités
    "verdict",  # décision déterministe SUFFISANT / PARTIEL / INSUFFISANT
    "notify",  # effet de bord (webhook, ticket)
]

TERMINAL_NODE_TYPE = "terminal"


class StrictModel(BaseModel):
    """Toute clé inconnue est une erreur : rien de « documentaire » en YAML."""

    model_config = ConfigDict(extra="forbid")


class FieldSpec(StrictModel):
    """Un champ du schéma d'entités (ex. ``commercial``, ``numero_reunion``)."""

    type: FieldType
    description: str = ""
    required: bool = False
    values: list[str] | None = Field(
        default=None, description="Valeurs permises, obligatoire pour le type enum"
    )

    @model_validator(mode="after")
    def _enum_needs_values(self) -> FieldSpec:
        if self.type == "enum" and not self.values:
            raise ValueError("un champ de type enum doit lister ses valeurs")
        if self.type != "enum" and self.values:
            raise ValueError("`values` n'a de sens que pour le type enum")
        return self


class EntitySchema(StrictModel):
    """Champs décrivant un dossier et un document pour ce cas d'usage."""

    dossier: dict[str, FieldSpec] = Field(default_factory=dict)
    document: dict[str, FieldSpec] = Field(default_factory=dict)


class ChunkingSpec(StrictModel):
    """Découpage, en **tokens** du modèle d'embedding (jamais en caractères).

    Un morceau « enfant » (``size_tokens``) est ce que l'on cherche et que
    l'on cite ; son « parent » (``parent_size_tokens``) est la section qui
    l'entoure, ce que l'on donne à lire au LLM. ``max_tokens`` est la limite
    haute absolue, préfixe de contexte compris : aucun morceau ne la dépasse.
    """

    size_tokens: int = Field(512, ge=64, le=4096)
    overlap_pct: int = Field(15, ge=0, le=50)
    parent_size_tokens: int = Field(1536, ge=64, le=8192)
    max_tokens: int = Field(2500, ge=64, le=8192)
    #: Tokenizer du modèle d'embedding : nom Hugging Face (lu dans le cache
    #: local, jamais téléchargé) ou chemin d'un ``tokenizer.json``.
    tokenizer: str = Field("BAAI/bge-m3", min_length=1)

    @model_validator(mode="after")
    def _ordered(self) -> ChunkingSpec:
        if not self.size_tokens <= self.parent_size_tokens <= self.max_tokens:
            raise ValueError(
                "chunking : size_tokens <= parent_size_tokens <= max_tokens attendu "
                f"({self.size_tokens}, {self.parent_size_tokens}, {self.max_tokens})"
            )
        return self

    @property
    def overlap_tokens(self) -> int:
        return self.size_tokens * self.overlap_pct // 100


class RetrievalSpec(StrictModel):
    dense_top_n: int = Field(75, ge=1)
    bm25_top_n: int = Field(75, ge=1)
    rrf_k: int = Field(60, ge=1)
    rerank_top_k: int = Field(8, ge=1)


class IngestionSpec(StrictModel):
    """Comment reconnaître les dossiers et lire les métadonnées par règles.

    - un dossier est un sous-répertoire de la racine ; ``dossier_file`` (s'il
      existe dedans) donne ses champs de dossier ;
    - ``filename_patterns`` sont des expressions régulières (insensibles à la
      casse) appliquées au nom de fichier ; chaque groupe nommé remplit le
      champ de document du même nom (ex. ``reunion-(?P<numero_reunion>\\d+)``) ;
    - ``document_order`` est le champ de document qui range les documents d'un
      dossier dans le temps (ex. ``numero_reunion``) : quand des documents
      donnent des valeurs différentes pour un champ de dossier, celle du
      document le plus récent l'emporte.
    """

    dossier_file: str = Field("dossier.yaml", pattern=r"^[^/\\]+$")
    filename_patterns: list[str] = Field(default_factory=list)
    document_order: str | None = None

    @model_validator(mode="after")
    def _patterns_compile(self) -> IngestionSpec:
        for pattern in self.filename_patterns:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                raise ValueError(f"expression régulière invalide `{pattern}` : {e}") from e
            if not compiled.groupindex:
                raise ValueError(f"`{pattern}` n'a aucun groupe nommé : il ne remplit aucun champ")
        return self

    def compiled_patterns(self) -> list[re.Pattern[str]]:
        return [re.compile(p, re.IGNORECASE) for p in self.filename_patterns]


class UseCaseProfile(StrictModel):
    """Contenu validé de ``profile.yaml``."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    title: str
    description: str = ""
    entities: EntitySchema = Field(default_factory=EntitySchema)
    ingestion: IngestionSpec = Field(default_factory=IngestionSpec)
    chunking: ChunkingSpec = Field(default_factory=ChunkingSpec)
    retrieval: RetrievalSpec = Field(default_factory=RetrievalSpec)

    @model_validator(mode="after")
    def _ingestion_matches_entities(self) -> UseCaseProfile:
        doc_fields = self.entities.document
        for pattern in self.ingestion.compiled_patterns():
            for group in pattern.groupindex:
                if group not in doc_fields:
                    raise ValueError(
                        f"le groupe `{group}` de `{pattern.pattern}` n'est pas un champ de "
                        f"document du profil ({', '.join(doc_fields) or 'aucun'})"
                    )
        order = self.ingestion.document_order
        if order is not None:
            if order not in doc_fields:
                raise ValueError(f"`document_order` : `{order}` n'est pas un champ de document")
            if doc_fields[order].type not in ("integer", "number", "date"):
                raise ValueError(f"`document_order` : `{order}` doit être un nombre ou une date")
        overlap = set(doc_fields) & set(self.entities.dossier)
        if overlap:
            raise ValueError(
                f"champ(s) à la fois de dossier et de document : {', '.join(sorted(overlap))}"
            )
        return self


class Rule(StrictModel):
    """Règle déterministe évaluée en code. ``when`` est une expression sur une
    liste blanche de champs ; sa grammaire est fixée à l'étape 5. Ici on exige
    seulement qu'elle soit non vide et qu'elle mène à un nœud existant."""

    when: str = Field(min_length=1)
    then: str = Field(min_length=1, description="Nom du nœud cible")
    reason: str = ""


class Node(StrictModel):
    """Un nœud de la boucle : soit un outil, soit un nœud terminal."""

    type: ToolName | Literal["terminal"]
    description: str = ""
    params: dict = Field(default_factory=dict)
    rules: list[Rule] = Field(
        default_factory=list, description="Évaluées dans l'ordre ; la première qui matche gagne"
    )
    default: str | None = Field(default=None, description="Nœud suivant si aucune règle ne matche")

    @model_validator(mode="after")
    def _terminal_has_no_exit(self) -> Node:
        if self.type == TERMINAL_NODE_TYPE and (self.rules or self.default):
            raise ValueError("un nœud terminal n'a ni règles ni suite : il n'est pas réentrant")
        if self.type != TERMINAL_NODE_TYPE and not self.rules and not self.default:
            raise ValueError("un nœud outil doit avoir au moins une règle ou un `default`")
        return self


class Loop(StrictModel):
    """Contenu validé de ``loop.yaml`` : une machine à états bornée."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    description: str = ""
    start: str
    max_steps: int = Field(
        6, ge=1, le=50, description="Plafond de nœuds traversés, décrémenté en code"
    )
    nodes: dict[str, Node]

    @model_validator(mode="after")
    def _graph_is_consistent(self) -> Loop:
        if self.start not in self.nodes:
            raise ValueError(f"le nœud de départ `{self.start}` n'existe pas")
        terminals = [n for n, spec in self.nodes.items() if spec.type == TERMINAL_NODE_TYPE]
        if not terminals:
            raise ValueError("une boucle doit avoir au moins un nœud terminal")
        for name, spec in self.nodes.items():
            targets = [r.then for r in spec.rules] + ([spec.default] if spec.default else [])
            for target in targets:
                if target not in self.nodes:
                    raise ValueError(f"le nœud `{name}` renvoie vers `{target}`, qui n'existe pas")
        return self

    def terminal_nodes(self) -> list[str]:
        return [n for n, spec in self.nodes.items() if spec.type == TERMINAL_NODE_TYPE]


class ProfileBundle(StrictModel):
    """Un cas d'usage chargé : son profil et sa boucle."""

    profile: UseCaseProfile
    loop: Loop
    path: Path

    @model_validator(mode="after")
    def _names_match(self) -> ProfileBundle:
        if self.profile.name != self.path.name:
            raise ValueError(
                f"le nom du profil `{self.profile.name}` doit être celui du dossier "
                f"`{self.path.name}`"
            )
        return self


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"fichier de profil absent : {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} doit contenir un objet YAML")
    return data


def load_profile(directory: str | Path) -> ProfileBundle:
    """Charge et valide ``configs/profiles/<cas>/``.

    Lève ``FileNotFoundError`` si un fichier manque et ``pydantic.ValidationError``
    si une clé est inconnue, un type faux ou le graphe de la boucle incohérent.
    """
    directory = Path(directory)
    profile = UseCaseProfile.model_validate(_read_yaml(directory / "profile.yaml"))
    loop = Loop.model_validate(_read_yaml(directory / "loop.yaml"))
    return ProfileBundle(profile=profile, loop=loop, path=directory)


def list_profiles(root: str | Path) -> list[str]:
    """Noms des cas d'usage présents sous ``root`` (un dossier par cas)."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "profile.yaml").is_file())
