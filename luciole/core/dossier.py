"""Dossiers et documents : identifiants stables et fiches de métadonnées.

Un **dossier** regroupe les documents d'une même affaire : un projet déposé
(cas MRAe : tomes, annexes, cartes) ou un client suivi par un commercial (cas
ventes : la suite des réunions). Sur disque, c'est un sous-répertoire de la
racine d'ingestion ; son identifiant est son chemin relatif à cette racine.

Un **document** est identifié par ``document_id`` = empreinte du chemin
relatif **et** du contenu. V5 utilisait le nom de fichier seul : deux
« Tome 4.pdf » de deux dossiers différents s'écrasaient. Ici :

- deux fichiers de même nom dans deux dossiers ont deux identifiants ;
- un fichier modifié change d'identifiant : l'ancien est remplacé, jamais
  dupliqué (le magasin garde un seul document par chemin) ;
- la même empreinte de contenu (``content_hash``) sert à vérifier un fichier.
"""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path, PurePath

from pydantic import BaseModel, ConfigDict, Field

from luciole.core.metadata import MetadataIssue
from luciole.core.versions import PARSER_VERSION

_HASH_BLOCK = 1 << 20


def content_hash(path: str | Path) -> str:
    """SHA-256 du contenu du fichier, lu par blocs (les études font 30 Mo)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(_HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def normalize_relative_path(path: str | PurePath) -> str:
    """Chemin relatif canonique : séparateurs ``/`` (Windows comme Linux),
    Unicode NFC (macOS écrit les accents en NFD), sans ``./`` initial."""
    posix = PurePath(path).as_posix() if not isinstance(path, str) else path.replace("\\", "/")
    parts = [p for p in posix.split("/") if p not in ("", ".")]
    if not parts or ".." in parts:
        raise ValueError(f"chemin relatif invalide : {path!r}")
    return unicodedata.normalize("NFC", "/".join(parts))


def make_document_id(relative_path: str, digest: str) -> str:
    """Identifiant d'un document : empreinte du chemin relatif et du contenu."""
    key = f"{normalize_relative_path(relative_path)}\n{digest}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


class DocumentRecord(BaseModel):
    """Fiche d'un document dans le magasin structuré."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    dossier_id: str
    relative_path: str = Field(description="Chemin relatif à la racine d'ingestion, en /")
    file_name: str
    doc_type: str
    content_hash: str
    size_bytes: int = Field(ge=0)
    page_count: int = Field(ge=0)
    parser_version: int = PARSER_VERSION
    metadata: dict[str, object] = Field(default_factory=dict)
    sources: dict[str, str] = Field(
        default_factory=dict, description="Origine de chaque valeur (front matter, nom…)"
    )
    issues: list[MetadataIssue] = Field(default_factory=list)
    #: Valeurs brutes des champs de **dossier** lues dans le document (front
    #: matter…), consolidées au niveau du dossier. Gardées dans la fiche pour
    #: reconsolider un dossier sans relire ses documents inchangés.
    dossier_hints: dict[str, object] = Field(default_factory=dict)

    @property
    def valid(self) -> bool:
        """Aucun problème bloquant (champ requis absent, valeur invalide)."""
        return not any(i.level == "error" for i in self.issues)


class DossierRecord(BaseModel):
    """Fiche d'un dossier dans le magasin structuré."""

    model_config = ConfigDict(extra="forbid")

    dossier_id: str
    metadata: dict[str, object] = Field(default_factory=dict)
    sources: dict[str, str] = Field(default_factory=dict)
    issues: list[MetadataIssue] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(i.level == "error" for i in self.issues)
