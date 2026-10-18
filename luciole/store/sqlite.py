"""Magasin structuré local (SQLite) des dossiers et documents.

Les boucles agentiques ne lisent pas tout : elles **sélectionnent** d'abord
(« les réunions des dossiers perdus de Camille Martin »), **comptent**
(« combien de dossiers gagnés par produit ») puis lisent les extraits utiles.
Ce magasin sert ces deux outils (``filter`` et ``aggregate``, étape 5) à
partir des fiches produites par l'ingestion.

Choix : SQLite de la bibliothèque standard (aucune dépendance, fichier
unique, fonctions JSON intégrées, identique sous Windows, Linux x86 et ARM).
Les métadonnées sont stockées en JSON et interrogées par ``json_extract`` ;
les noms de champs sont **validés contre le profil** et passés en paramètres,
jamais concaténés dans la requête.

Un magasin appartient à un cas d'usage : l'ouvrir avec un autre profil est
refusé. Il garde **un seul document par chemin** : un fichier modifié
remplace sa fiche (``replaced``) et le rapport donne l'ancien identifiant,
pour que l'index (étape 3) supprime ses anciens chunks.

Il garde aussi les **morceaux** de chaque document (parents et enfants) :
l'index de recherche ne contient que les enfants, la lecture d'une section
entière passe par ici. ``indexed_with`` note avec quel embedder et quel
schéma d'index un document a été indexé : un document dont la valeur ne
correspond plus est réindexé.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from luciole.core.chunk import Chunk
from luciole.core.dossier import DocumentRecord, DossierRecord
from luciole.core.metadata import MetadataIssue, coerce_value
from luciole.core.profile import FieldSpec, UseCaseProfile
from luciole.core.versions import STORE_SCHEMA_VERSION

if TYPE_CHECKING:
    from luciole.ingestion.dossiers import ScanResult

Level = Literal["dossier", "document"]
Op = Literal["eq", "ne", "lt", "le", "gt", "ge", "in"]

_SQL_OPS = {"eq": "=", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
_FIELD_RE = re.compile(r"^[a-z_][a-z0-9_]*$")

# Colonnes fixes interrogeables comme des champs (en plus des métadonnées).
_DOCUMENT_COLUMNS = {"dossier_id", "doc_type", "file_name", "relative_path", "page_count"}
_DOSSIER_COLUMNS = {"dossier_id"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS dossiers (
    dossier_id TEXT PRIMARY KEY,
    metadata   TEXT NOT NULL,
    sources    TEXT NOT NULL,
    issues     TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    document_id    TEXT PRIMARY KEY,
    dossier_id     TEXT NOT NULL REFERENCES dossiers(dossier_id) ON DELETE CASCADE,
    relative_path  TEXT NOT NULL UNIQUE,
    file_name      TEXT NOT NULL,
    doc_type       TEXT NOT NULL,
    content_hash   TEXT NOT NULL,
    size_bytes     INTEGER NOT NULL,
    page_count     INTEGER NOT NULL,
    parser_version INTEGER NOT NULL,
    metadata       TEXT NOT NULL,
    sources        TEXT NOT NULL,
    issues         TEXT NOT NULL,
    ingested_at    TEXT NOT NULL,
    indexed_with   TEXT
);
CREATE INDEX IF NOT EXISTS documents_by_dossier ON documents(dossier_id);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,
    parent_id   TEXT,
    position    INTEGER NOT NULL,
    data        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_document ON chunks(document_id, kind, position);
"""


@dataclass(frozen=True)
class Condition:
    """Une condition de filtre : ``champ`` ``op`` ``valeur``. Au niveau
    document, ``dossier.<champ>`` désigne un champ du dossier parent."""

    field: str
    op: Op
    value: object


@dataclass
class SyncReport:
    """Ce qu'une synchronisation a changé (utile à l'index de l'étape 3)."""

    new: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    replaced: dict[str, str] = field(default_factory=dict)  # nouvel id -> ancien id
    removed: list[str] = field(default_factory=list)
    removed_dossiers: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{len(self.new)} nouveau(x), {len(self.unchanged)} inchangé(s), "
            f"{len(self.replaced)} remplacé(s), {len(self.removed)} supprimé(s), "
            f"{len(self.removed_dossiers)} dossier(s) supprimé(s)"
        )


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _issues_dump(issues: list[MetadataIssue]) -> str:
    return _dumps([i.model_dump() for i in issues])


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


class MetadataStore:
    """Fiches de dossiers et de documents d'un cas d'usage."""

    def __init__(self, path: str | Path, profile: UseCaseProfile) -> None:
        self.profile = profile
        self.path = str(path)
        self._db = sqlite3.connect(self.path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self._db.execute("PRAGMA journal_mode = WAL")
        self._db.executescript(_SCHEMA)
        self._check_identity()

    # -- cycle de vie -----------------------------------------------------
    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> MetadataStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _check_identity(self) -> None:
        rows = dict(self._db.execute("SELECT key, value FROM store_meta").fetchall())
        if not rows:
            with self._db:
                self._db.executemany(
                    "INSERT INTO store_meta(key, value) VALUES (?, ?)",
                    [("profile", self.profile.name), ("schema_version", str(STORE_SCHEMA_VERSION))],
                )
            return
        if rows.get("profile") != self.profile.name:
            raise ValueError(
                f"ce magasin appartient au cas d'usage « {rows.get('profile')} », "
                f"pas à « {self.profile.name} »"
            )
        if rows.get("schema_version") != str(STORE_SCHEMA_VERSION):
            raise ValueError(
                f"magasin en version {rows.get('schema_version')}, attendu "
                f"{STORE_SCHEMA_VERSION} : le reconstruire par une nouvelle ingestion"
            )

    # -- écriture ---------------------------------------------------------
    def upsert_dossier(self, dossier: DossierRecord) -> None:
        with self._db:
            self._db.execute(
                """INSERT INTO dossiers(dossier_id, metadata, sources, issues, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(dossier_id) DO UPDATE SET metadata = excluded.metadata,
                     sources = excluded.sources, issues = excluded.issues,
                     updated_at = excluded.updated_at""",
                (
                    dossier.dossier_id,
                    _dumps(dossier.metadata),
                    _dumps(dossier.sources),
                    _issues_dump(dossier.issues),
                    _now(),
                ),
            )

    def upsert_document(self, doc: DocumentRecord) -> tuple[str, str | None]:
        """Enregistre une fiche ; renvoie ``(statut, ancien_id)`` avec statut
        ``new``, ``unchanged`` (même contenu au même chemin) ou ``replaced``."""
        previous = self._db.execute(
            "SELECT document_id FROM documents WHERE relative_path = ?", (doc.relative_path,)
        ).fetchone()
        status, old_id = "new", None
        if previous is not None:
            old_id = previous["document_id"]
            status = "unchanged" if old_id == doc.document_id else "replaced"
        with self._db:
            if status == "replaced":
                self._db.execute("DELETE FROM documents WHERE document_id = ?", (old_id,))
            self._db.execute(
                """INSERT INTO documents(document_id, dossier_id, relative_path, file_name,
                     doc_type, content_hash, size_bytes, page_count, parser_version,
                     metadata, sources, issues, ingested_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(document_id) DO UPDATE SET dossier_id = excluded.dossier_id,
                     doc_type = excluded.doc_type, page_count = excluded.page_count,
                     parser_version = excluded.parser_version, metadata = excluded.metadata,
                     sources = excluded.sources, issues = excluded.issues""",
                (
                    doc.document_id,
                    doc.dossier_id,
                    doc.relative_path,
                    doc.file_name,
                    doc.doc_type,
                    doc.content_hash,
                    doc.size_bytes,
                    doc.page_count,
                    doc.parser_version,
                    _dumps(doc.metadata),
                    _dumps(doc.sources),
                    _issues_dump(doc.issues),
                    _now(),
                ),
            )
        return status, (old_id if status == "replaced" else None)

    def sync(self, result: ScanResult) -> SyncReport:
        """Aligne le magasin sur un parcours complet de la racine : fiches
        ajoutées ou remplacées, documents et dossiers disparus supprimés."""
        report = SyncReport()
        for dossier in result.dossiers:
            self.upsert_dossier(dossier)
        seen: set[str] = set()
        for doc in result.documents:
            status, old_id = self.upsert_document(doc)
            seen.add(doc.document_id)
            if status == "replaced":
                report.replaced[doc.document_id] = old_id
            else:
                getattr(report, status).append(doc.document_id)
        stale = [
            r["document_id"]
            for r in self._db.execute("SELECT document_id FROM documents")
            if r["document_id"] not in seen
        ]
        kept_dossiers = {d.dossier_id for d in result.dossiers}
        gone = [
            r["dossier_id"]
            for r in self._db.execute("SELECT dossier_id FROM dossiers")
            if r["dossier_id"] not in kept_dossiers
        ]
        with self._db:
            self._db.executemany(
                "DELETE FROM documents WHERE document_id = ?", [(i,) for i in stale]
            )
            self._db.executemany("DELETE FROM dossiers WHERE dossier_id = ?", [(i,) for i in gone])
        report.removed = sorted(stale)
        report.removed_dossiers = sorted(gone)
        return report

    # -- morceaux et état d'indexation -------------------------------------
    def replace_chunks(self, document_id: str, chunks: list[Chunk], indexed_with: str) -> None:
        """Remplace les morceaux d'un document et note la clé d'indexation.
        Appelé **après** l'écriture dans l'index : en cas d'arrêt entre les deux,
        le document est simplement réindexé au passage suivant."""
        if self.get_document(document_id) is None:
            raise KeyError(f"document inconnu du magasin : {document_id}")
        with self._db:
            self._db.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            self._db.executemany(
                """INSERT INTO chunks(chunk_id, document_id, kind, parent_id, position, data)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    (c.chunk_id, document_id, c.kind, c.parent_id, c.index, c.model_dump_json())
                    for c in chunks
                ],
            )
            self._db.execute(
                "UPDATE documents SET indexed_with = ? WHERE document_id = ?",
                (indexed_with, document_id),
            )

    def indexed_with(self, document_id: str) -> str | None:
        row = self._db.execute(
            "SELECT indexed_with FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        return row["indexed_with"] if row else None

    def chunks_of(self, document_id: str, kind: str | None = None) -> list[Chunk]:
        sql = "SELECT data FROM chunks WHERE document_id = ?"
        params: list = [document_id]
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY kind DESC, position"  # parents puis enfants
        return [Chunk.model_validate_json(r["data"]) for r in self._db.execute(sql, params)]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        row = self._db.execute("SELECT data FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        return Chunk.model_validate_json(row["data"]) if row else None

    def count_chunks(self, kind: str | None = None) -> int:
        if kind is None:
            return self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return self._db.execute("SELECT COUNT(*) FROM chunks WHERE kind = ?", (kind,)).fetchone()[0]

    # -- lecture ----------------------------------------------------------
    @staticmethod
    def _dossier(row: sqlite3.Row) -> DossierRecord:
        return DossierRecord(
            dossier_id=row["dossier_id"],
            metadata=json.loads(row["metadata"]),
            sources=json.loads(row["sources"]),
            issues=[MetadataIssue(**i) for i in json.loads(row["issues"])],
        )

    @staticmethod
    def _document(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            document_id=row["document_id"],
            dossier_id=row["dossier_id"],
            relative_path=row["relative_path"],
            file_name=row["file_name"],
            doc_type=row["doc_type"],
            content_hash=row["content_hash"],
            size_bytes=row["size_bytes"],
            page_count=row["page_count"],
            parser_version=row["parser_version"],
            metadata=json.loads(row["metadata"]),
            sources=json.loads(row["sources"]),
            issues=[MetadataIssue(**i) for i in json.loads(row["issues"])],
        )

    def get_document(self, document_id: str) -> DocumentRecord | None:
        row = self._db.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        return self._document(row) if row else None

    def get_dossier(self, dossier_id: str) -> DossierRecord | None:
        row = self._db.execute(
            "SELECT * FROM dossiers WHERE dossier_id = ?", (dossier_id,)
        ).fetchone()
        return self._dossier(row) if row else None

    # -- requêtes (outils filter / aggregate) ------------------------------
    def _resolve(self, level: Level, name: str) -> tuple[str, list, FieldSpec | None]:
        """Expression SQL d'un champ, ses paramètres et sa spécification."""
        target, alias, columns = level, "x", _DOCUMENT_COLUMNS
        if level == "dossier":
            columns = _DOSSIER_COLUMNS
        if level == "document" and name.startswith("dossier."):
            target, alias, columns, name = "dossier", "s", _DOSSIER_COLUMNS, name[len("dossier.") :]
        if not _FIELD_RE.match(name):
            raise ValueError(f"nom de champ invalide : {name!r}")
        schema = (
            self.profile.entities.dossier if target == "dossier" else self.profile.entities.document
        )
        if name in schema:
            return f"json_extract({alias}.metadata, ?)", [f"$.{name}"], schema[name]
        if name in columns:
            return f"{alias}.{name}", [], None
        known = sorted({*schema, *columns})
        raise ValueError(f"champ inconnu au niveau {target} : {name} (connus : {', '.join(known)})")

    @staticmethod
    def _param(spec: FieldSpec | None, value: object) -> object:
        if spec is None:
            return value
        canonical = coerce_value(spec, value)
        return int(canonical) if isinstance(canonical, bool) else canonical

    def _where(self, level: Level, conditions: list[Condition]) -> tuple[str, list]:
        clauses, params = [], []
        for c in conditions:
            expr, expr_params, spec = self._resolve(level, c.field)
            if c.op == "in":
                values = list(c.value) if isinstance(c.value, list | tuple | set) else [c.value]
                if not values:
                    clauses.append("0")
                    continue
                clauses.append(f"{expr} IN ({', '.join('?' * len(values))})")
                params += expr_params + [self._param(spec, v) for v in values]
            elif c.op in _SQL_OPS:
                clauses.append(f"{expr} {_SQL_OPS[c.op]} ?")
                params += expr_params + [self._param(spec, c.value)]
            else:
                raise ValueError(f"opérateur inconnu : {c.op}")
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def _from(self, level: Level) -> str:
        if level == "dossier":
            return "FROM dossiers x"
        return "FROM documents x JOIN dossiers s ON s.dossier_id = x.dossier_id"

    def _order(self, level: Level) -> tuple[str, list]:
        if level == "dossier":
            return " ORDER BY x.dossier_id", []
        order = self.profile.ingestion.document_order
        if order:
            return " ORDER BY x.dossier_id, json_extract(x.metadata, ?), x.relative_path", [
                f"$.{order}"
            ]
        return " ORDER BY x.dossier_id, x.relative_path", []

    def select(
        self, level: Level, conditions: list[Condition] | None = None
    ) -> list[DossierRecord] | list[DocumentRecord]:
        """Fiches qui satisfont toutes les conditions ; les documents sortent
        par dossier puis dans l'ordre chronologique du profil."""
        where, params = self._where(level, conditions or [])
        order, order_params = self._order(level)
        rows = self._db.execute(
            f"SELECT x.* {self._from(level)}{where}{order}", params + order_params
        ).fetchall()
        build = self._dossier if level == "dossier" else self._document
        return [build(r) for r in rows]

    def count_by(
        self, level: Level, group_field: str, conditions: list[Condition] | None = None
    ) -> dict[object, int]:
        """Nombre de fiches par valeur de ``group_field`` (``None`` = sans valeur)."""
        expr, expr_params, _ = self._resolve(level, group_field)
        where, params = self._where(level, conditions or [])
        rows = self._db.execute(
            f"SELECT {expr} AS v, COUNT(*) AS n {self._from(level)}{where} GROUP BY v ORDER BY v",
            expr_params + params,
        ).fetchall()
        return {r["v"]: r["n"] for r in rows}
