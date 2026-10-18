"""Découverte des dossiers et lecture de leurs métadonnées par règles.

Organisation attendue sur disque (identique pour tous les cas d'usage)::

    <racine>/
      <dossier A>/            ← un dossier (projet, client…)
        dossier.yaml          ← optionnel : champs du dossier
        reunion-1.transcript.md
        Annexes/carte.pdf     ← les sous-répertoires appartiennent au dossier
      <dossier B>/
        ...

Métadonnées d'un **document**, par priorité décroissante :

1. ce que le document dit de lui-même (front matter YAML, en-têtes d'un
   courriel…), tel que le parser l'a lu ;
2. les ``filename_patterns`` du profil appliqués au nom de fichier ;
3. un extracteur optionnel (LLM local à sortie JSON validée, branché à
   l'étape 4) pour les seuls champs encore manquants.

Métadonnées d'un **dossier** : ``dossier.yaml`` d'abord, sinon ce que les
documents en disent. S'ils divergent (l'issue d'un dossier commercial passe
de « en cours » à « perdu » au fil des réunions), la valeur du document le
plus récent selon ``document_order`` l'emporte, avec un avertissement.

Toute valeur est convertie et validée contre le schéma du profil
(``luciole.core.metadata``) ; rien n'est deviné. Un fichier illisible ou un
champ invalide n'arrête pas l'ingestion : il apparaît dans le rapport.
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import yaml
from loguru import logger

from luciole.core.document import ParsedDocument
from luciole.core.dossier import (
    DocumentRecord,
    DossierRecord,
    content_hash,
    make_document_id,
    normalize_relative_path,
)
from luciole.core.metadata import MetadataIssue, ValidatedMetadata, coerce_value, merge_sources
from luciole.core.profile import FieldSpec, UseCaseProfile, load_profile

# Fichiers qui ne sont jamais des documents (verrous Office, vignettes…).
_IGNORED_NAMES = {"thumbs.db", "desktop.ini", ".ds_store"}
_IGNORED_PREFIXES = (".", "~$")

SOURCE_DOCUMENT = "document"
SOURCE_FILENAME = "nom_de_fichier"
SOURCE_EXTRACTOR = "extracteur"
SOURCE_DOSSIER_FILE = "fichier_dossier"
SOURCE_DOCUMENTS = "documents"


class Registry(Protocol):
    def supports(self, path: str | Path) -> bool: ...
    def parse(self, path: str | Path) -> ParsedDocument: ...


class MetadataExtractor(Protocol):
    """Extraction des champs manquants par un modèle (étape 4 : LLM local à
    sortie JSON). Renvoie des valeurs brutes, validées ensuite comme les
    autres ; une valeur invalide est refusée, jamais corrigée en silence."""

    name: str

    def extract(self, document: ParsedDocument, fields: dict[str, FieldSpec]) -> dict: ...


@dataclass(frozen=True)
class SkippedFile:
    relative_path: str
    reason: str


@dataclass
class ScanResult:
    """Ce qu'une ingestion a trouvé sous la racine."""

    root: Path
    dossiers: list[DossierRecord] = field(default_factory=list)
    documents: list[DocumentRecord] = field(default_factory=list)
    skipped: list[SkippedFile] = field(default_factory=list)

    def documents_of(self, dossier_id: str) -> list[DocumentRecord]:
        return [d for d in self.documents if d.dossier_id == dossier_id]


# ---------------------------------------------------------------------------
# Règles
# ---------------------------------------------------------------------------
def filename_values(profile: UseCaseProfile, file_name: str) -> dict[str, str]:
    """Champs lus dans le nom de fichier par les motifs du profil (le premier
    motif qui donne un champ l'emporte)."""
    values: dict[str, str] = {}
    file_name = unicodedata.normalize("NFC", file_name)  # macOS écrit les accents en NFD
    for pattern in profile.ingestion.compiled_patterns():
        m = pattern.search(file_name)
        if not m:
            continue
        for group, value in m.groupdict().items():
            if value is not None and group not in values:
                values[group] = value
    return values


def document_metadata(
    profile: UseCaseProfile,
    parsed: ParsedDocument,
    file_name: str,
    extractor: MetadataExtractor | None = None,
) -> ValidatedMetadata:
    schema = profile.entities.document
    candidates: list[tuple[str, dict]] = [
        (SOURCE_DOCUMENT, parsed.metadata),
        (SOURCE_FILENAME, filename_values(profile, file_name)),
    ]
    result = merge_sources(schema, candidates)
    if extractor is None:
        return result
    wanted = {name: spec for name, spec in schema.items() if name not in result.values}
    if not wanted:
        return result
    try:
        extracted = extractor.extract(parsed, wanted) or {}
    except Exception as e:  # noqa: BLE001 - un extracteur en échec ne bloque pas
        logger.warning(f"{file_name} : extracteur {extractor.name} en échec ({e})")
        result.issues.append(
            MetadataIssue(
                level="warning",
                field="*",
                message=f"extracteur en échec : {e}",
                source=SOURCE_EXTRACTOR,
            )
        )
        return result
    return merge_sources(schema, [*candidates, (SOURCE_EXTRACTOR, extracted)])


def _read_dossier_file(path: Path, profile: UseCaseProfile) -> tuple[dict, list[MetadataIssue]]:
    if not path.is_file():
        return {}, []
    issues: list[MetadataIssue] = []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    except (yaml.YAMLError, UnicodeDecodeError) as e:
        return {}, [
            MetadataIssue(
                level="error", field="*", message=f"{path.name} illisible : {e}",
                source=SOURCE_DOSSIER_FILE,
            )
        ]  # fmt: skip
    if not isinstance(data, dict):
        return {}, [
            MetadataIssue(
                level="error", field="*", message=f"{path.name} doit être un objet YAML",
                source=SOURCE_DOSSIER_FILE,
            )
        ]  # fmt: skip
    known = profile.entities.dossier
    for key in data:
        if key not in known:
            issues.append(
                MetadataIssue(
                    level="warning",
                    field=str(key),
                    message=f"champ inconnu du profil « {profile.name} » ({', '.join(known)})",
                    source=SOURCE_DOSSIER_FILE,
                )
            )
    return data, issues


def _order_key(profile: UseCaseProfile, doc: DocumentRecord) -> tuple[int, object, str]:
    order = profile.ingestion.document_order
    value = doc.metadata.get(order) if order else None
    # Les documents sans valeur d'ordre passent avant les autres.
    return (0, 0, doc.relative_path) if value is None else (1, value, doc.relative_path)


def dossier_metadata(
    profile: UseCaseProfile,
    dossier_id: str,
    dossier_file_values: dict,
    documents: list[tuple[DocumentRecord, ParsedDocument]],
) -> ValidatedMetadata:
    """Champs du dossier : ``dossier.yaml``, sinon consolidés depuis les documents."""
    schema = profile.entities.dossier
    from_documents: dict[str, object] = {}
    issues: list[MetadataIssue] = []
    ordered = sorted(documents, key=lambda pair: _order_key(profile, pair[0]))
    for name, spec in schema.items():
        seen: list[tuple[str, object]] = []
        for record, parsed in ordered:
            if name not in parsed.metadata:
                continue
            try:
                value = coerce_value(spec, parsed.metadata[name])
            except ValueError as e:
                issues.append(
                    MetadataIssue(
                        level="warning", field=name,
                        message=f"{record.relative_path} : {e}", source=SOURCE_DOCUMENTS,
                    )
                )  # fmt: skip
                continue
            if value is not None:
                seen.append((record.relative_path, value))
        distinct = {v for _, v in seen}
        if not distinct:
            continue
        if len(distinct) > 1:
            if name in dossier_file_values:
                continue  # dossier.yaml tranche, pas besoin d'avertir
            if profile.ingestion.document_order is None:
                issues.append(
                    MetadataIssue(
                        level="error", field=name, source=SOURCE_DOCUMENTS,
                        message=(
                            "valeurs différentes selon les documents ("
                            + ", ".join(f"{p} : {v}" for p, v in seen)
                            + ") et aucun `document_order` pour départager"
                        ),
                    )
                )  # fmt: skip
                continue
            path, value = seen[-1]
            listed = ", ".join(sorted(map(str, distinct)))
            issues.append(
                MetadataIssue(
                    level="warning", field=name, source=SOURCE_DOCUMENTS,
                    message=(
                        f"valeurs différentes selon les documents ({listed}) ;"
                        f" retenue : {value} (document le plus récent, {path})"
                    ),
                )
            )  # fmt: skip
            from_documents[name] = value
        else:
            from_documents[name] = seen[-1][1]
    result = merge_sources(
        schema, [(SOURCE_DOSSIER_FILE, dossier_file_values), (SOURCE_DOCUMENTS, from_documents)]
    )
    # Les problèmes de consolidation restent visibles, sauf pour un champ
    # qui n'a finalement pas de valeur (déjà signalé comme manquant).
    result.issues = [
        *[i for i in issues if i.field in result.values or i.level == "error"],
        *result.issues,
    ]
    return result


# ---------------------------------------------------------------------------
# Parcours
# ---------------------------------------------------------------------------
def _is_ignored(path: Path) -> bool:
    name = path.name
    return name.lower() in _IGNORED_NAMES or name.startswith(_IGNORED_PREFIXES)


def _iter_files(directory: Path) -> Iterator[Path]:
    for path in sorted(directory.rglob("*")):
        if path.is_file() and not any(_is_ignored(p) for p in path.relative_to(directory).parents):
            if not _is_ignored(path):
                yield path


def scan(
    root: str | Path,
    profile: UseCaseProfile,
    registry: Registry,
    *,
    extractor: MetadataExtractor | None = None,
    on_document: Callable[[DocumentRecord, ParsedDocument], None] | None = None,
) -> ScanResult:
    """Parcourt ``root`` : un sous-répertoire = un dossier.

    Chaque document est lu une seule fois ; ``on_document`` reçoit sa fiche et
    sa version structurée (l'étape 3 y branchera découpage et indexation sans
    relire le fichier). Les fiches ne gardent que les métadonnées.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    result = ScanResult(root=root)
    dossier_file = profile.ingestion.dossier_file

    for entry in sorted(root.iterdir()):
        if _is_ignored(entry):
            continue
        rel = normalize_relative_path(entry.relative_to(root))
        if entry.is_file():
            result.skipped.append(SkippedFile(rel, "fichier hors de tout dossier (à la racine)"))
            continue
        if not entry.is_dir():
            continue
        dossier_id = rel
        parsed_docs: list[tuple[DocumentRecord, ParsedDocument]] = []
        for path in _iter_files(entry):
            rel_path = normalize_relative_path(path.relative_to(root))
            if path.parent == entry and path.name == dossier_file:
                continue
            if not registry.supports(path):
                result.skipped.append(SkippedFile(rel_path, "format non pris en charge"))
                continue
            try:
                parsed = registry.parse(path)
            except Exception as e:  # noqa: BLE001 - un fichier illisible n'arrête rien
                logger.warning(f"{rel_path} : lecture impossible ({e})")
                result.skipped.append(SkippedFile(rel_path, f"lecture impossible : {e}"))
                continue
            digest = content_hash(path)
            meta = document_metadata(profile, parsed, path.name, extractor)
            record = DocumentRecord(
                document_id=make_document_id(rel_path, digest),
                dossier_id=dossier_id,
                relative_path=rel_path,
                file_name=path.name,
                doc_type=parsed.doc_type,
                content_hash=digest,
                size_bytes=path.stat().st_size,
                page_count=len(parsed.pages),
                parser_version=parsed.parser_version,
                metadata=meta.values,
                sources=meta.sources,
                issues=meta.issues,
            )
            parsed_docs.append((record, parsed))
            if on_document is not None:
                on_document(record, parsed)

        file_values, file_issues = _read_dossier_file(entry / dossier_file, profile)
        meta = dossier_metadata(profile, dossier_id, file_values, parsed_docs)
        result.dossiers.append(
            DossierRecord(
                dossier_id=dossier_id,
                metadata=meta.values,
                sources=meta.sources,
                issues=[*file_issues, *meta.issues],
            )
        )
        result.documents.extend(record for record, _ in parsed_docs)
    return result


# ---------------------------------------------------------------------------
# Ligne de commande : rapport d'ingestion lisible
# ---------------------------------------------------------------------------
def format_report(result: ScanResult) -> str:
    lines = [f"Racine : {result.root}", ""]
    for dossier in result.dossiers:
        docs = result.documents_of(dossier.dossier_id)
        flag = "OK " if dossier.valid and all(d.valid for d in docs) else "!! "
        lines.append(f"{flag}Dossier « {dossier.dossier_id} » : {len(docs)} document(s)")
        for name, value in dossier.metadata.items():
            lines.append(f"     {name} = {value}   [{dossier.sources[name]}]")
        lines += [f"     {i.level.upper()} {i.field} : {i.message}" for i in dossier.issues]
        for doc in docs:
            mark = "  " if doc.valid else "!!"
            meta = ", ".join(f"{k}={v}" for k, v in doc.metadata.items()) or "aucun champ"
            lines.append(
                f"   {mark} {doc.relative_path} ({doc.doc_type}, {doc.page_count} p.) {meta}"
            )
            lines += [f"        {i.level.upper()} {i.field} : {i.message}" for i in doc.issues]
        lines.append("")
    if result.skipped:
        lines.append(f"Fichiers écartés : {len(result.skipped)}")
        lines += [f"   {s.relative_path} : {s.reason}" for s in result.skipped]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from luciole.ingestion.registry import default_registry

    parser = argparse.ArgumentParser(
        prog="python -m luciole.ingestion.dossiers",
        description="Lit une racine de dossiers et affiche les métadonnées trouvées.",
    )
    parser.add_argument("root", help="dossier racine (un sous-répertoire par dossier)")
    parser.add_argument("--profile", required=True, help="nom du cas d'usage (mrae, ventes…)")
    parser.add_argument("--profiles-dir", default="configs/profiles")
    parser.add_argument("--store", help="fichier SQLite où enregistrer le résultat (optionnel)")
    args = parser.parse_args(argv)

    bundle = load_profile(Path(args.profiles_dir) / args.profile)
    result = scan(args.root, bundle.profile, default_registry())
    print(format_report(result))
    if args.store:
        from luciole.store.sqlite import MetadataStore

        with MetadataStore(args.store, bundle.profile) as store:
            report = store.sync(result)
        print(f"\nMagasin {args.store} : {report.summary()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
