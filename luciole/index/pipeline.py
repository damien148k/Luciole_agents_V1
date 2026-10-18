"""Pipeline d'ingestion : une racine de dossiers → magasin + index alignés.

Un passage (``run``) :

1. parcourt la racine (``scan``), chaque fichier lu une seule fois ;
2. pour chaque document **nouveau, modifié, ou indexé avec une autre clé**
   (autre embedder, autre schéma) : découpe, calcule les vecteurs des
   enfants (par lots), puis **remplace** ses points dans Qdrant et OpenSearch
   (suppression par ``document_id`` puis écriture : pas de doublon) ;
3. synchronise le magasin (fiches ajoutées, remplacées, supprimées), puis y
   enregistre les morceaux et la clé d'indexation des documents traités ;
4. supprime de l'index les documents disparus et les anciennes versions des
   documents modifiés.

L'index est écrit **avant** le magasin : si le passage s'arrête au milieu,
le magasin ne note pas le document comme indexé et le passage suivant le
refait. Les documents inchangés ne sont ni redécoupés ni réembarqués.

En ligne de commande (services de ``docker/compose.services.yaml`` lancés) :

    python -m luciole.index.pipeline <racine> --profile ventes --hardware cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from luciole.core.chunk import Chunk
from luciole.core.document import ParsedDocument
from luciole.core.dossier import DocumentRecord
from luciole.core.profile import UseCaseProfile
from luciole.index.embedder import Embedder, index_key
from luciole.index.opensearch import OpenSearchIndex
from luciole.index.qdrant import QdrantIndex
from luciole.ingestion.chunking import Chunker
from luciole.ingestion.dossiers import Registry, SkippedFile, scan
from luciole.store.sqlite import MetadataStore, SyncReport


@dataclass
class PipelineReport:
    sync: SyncReport
    indexed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # déjà indexés avec la même clé
    deleted: list[str] = field(default_factory=list)  # retirés de l'index
    ignored: list[SkippedFile] = field(default_factory=list)  # illisibles, hors format
    children: int = 0
    parents: int = 0
    seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"{len(self.indexed)} document(s) indexé(s) ({self.children} morceaux enfants, "
            f"{self.parents} parents), {len(self.skipped)} déjà à jour, "
            f"{len(self.deleted)} retiré(s) de l'index, en {self.seconds:.1f} s"
        )


class IngestionPipeline:
    def __init__(
        self,
        profile: UseCaseProfile,
        store: MetadataStore,
        registry: Registry,
        chunker: Chunker,
        embedder: Embedder,
        vectors: QdrantIndex,
        lexical: OpenSearchIndex | None = None,
        *,
        batch_size: int = 16,
    ) -> None:
        if vectors.dimension != embedder.dimension:
            raise ValueError(
                f"dimension de l'index ({vectors.dimension}) ≠ embedder ({embedder.dimension})"
            )
        self.profile = profile
        self.store = store
        self.registry = registry
        self.chunker = chunker
        self.embedder = embedder
        self.vectors = vectors
        self.lexical = lexical
        self.batch_size = max(1, batch_size)
        self.key = index_key(embedder)

    # ------------------------------------------------------------------
    def run(self, root: str | Path) -> PipelineReport:
        start = time.perf_counter()
        self.vectors.ensure()
        if self.lexical is not None:
            self.lexical.ensure()
        pending: dict[str, list[Chunk]] = {}
        skipped: list[str] = []

        def on_document(record: DocumentRecord, parsed: ParsedDocument) -> None:
            if self.store.indexed_with(record.document_id) == self.key:
                skipped.append(record.document_id)
                return
            pending[record.document_id] = self.index_document(record, parsed)

        result = scan(root, self.profile, self.registry, on_document=on_document)
        sync = self.store.sync(result)
        for document_id, chunks in pending.items():
            self.store.replace_chunks(document_id, chunks, self.key)
        deleted = sorted({*sync.removed, *sync.replaced.values()})
        for document_id in deleted:
            self.delete_document(document_id)
        report = PipelineReport(
            sync=sync,
            indexed=sorted(pending),
            skipped=sorted(skipped),
            deleted=deleted,
            ignored=list(result.skipped),
            children=sum(1 for cs in pending.values() for c in cs if c.kind == "child"),
            parents=sum(1 for cs in pending.values() for c in cs if c.kind == "parent"),
            seconds=time.perf_counter() - start,
        )
        logger.info(report.summary())
        return report

    def index_document(self, record: DocumentRecord, parsed: ParsedDocument) -> list[Chunk]:
        """Découpe et indexe un document ; renvoie ses morceaux (pour le magasin)."""
        chunks = self.chunker.chunk(parsed, record)
        children = [c for c in chunks if c.kind == "child"]
        vectors = {
            "content": self._embed([c.text for c in children]),
            "context": self._embed([c.embed_text for c in children]),
        }
        self.vectors.replace_document(record.document_id, children, vectors)
        if self.lexical is not None:
            self.lexical.replace_document(record.document_id, children)
        logger.debug(f"{record.relative_path} : {len(children)} enfant(s) indexé(s)")
        return chunks

    def delete_document(self, document_id: str) -> None:
        self.vectors.delete_document(document_id)
        if self.lexical is not None:
            self.lexical.delete_document(document_id)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.embedder.embed(texts[i : i + self.batch_size])
            if len(batch) != len(texts[i : i + self.batch_size]):
                raise ValueError("l'embedder a renvoyé un nombre de vecteurs inattendu")
            for vector in batch:
                if len(vector) != self.embedder.dimension:
                    raise ValueError(
                        f"vecteur de dimension {len(vector)}, attendu {self.embedder.dimension}"
                    )
            out.extend(batch)
        return out


# ---------------------------------------------------------------------------
# Ligne de commande : indexer une racine de dossiers
# ---------------------------------------------------------------------------
def _hardware(name: str | None, root: Path):
    from luciole.core.hardware import (
        detect_arch,
        detect_gpu_memory_gb,
        list_hardware_profiles,
        suggest_profile,
    )

    profiles = list_hardware_profiles(root)
    if name is None:
        name = suggest_profile(profiles, detect_arch(), detect_gpu_memory_gb())
        if name is None:
            raise SystemExit("aucun profil matériel ne convient : donner --hardware")
    if name not in profiles:
        raise SystemExit(f"profil matériel « {name} » inconnu ({', '.join(profiles)})")
    return profiles[name]


def main(
    argv: list[str] | None = None,
    *,
    embedder_factory: Callable[..., Embedder] | None = None,
    qdrant=None,
    opensearch=None,
) -> int:
    """``embedder_factory(spec, hardware)`` et les clients remplacent le vrai
    modèle et les vrais services dans les tests."""
    from luciole.core.model_files import load_manifest
    from luciole.core.profile import load_profile
    from luciole.index.clients import opensearch_client, qdrant_client
    from luciole.index.embedder import index_name
    from luciole.ingestion.registry import default_registry
    from luciole.ingestion.tokenizer import load_tokenizer

    parser = argparse.ArgumentParser(
        prog="python -m luciole.index.pipeline",
        description="Indexe une racine de dossiers (Qdrant + OpenSearch) ; "
        "les fichiers inchangés ne sont pas recalculés.",
    )
    parser.add_argument("root", help="dossier racine (un sous-répertoire par dossier)")
    parser.add_argument("--profile", required=True, help="nom du cas d'usage (mrae, ventes…)")
    parser.add_argument("--profiles-dir", default="configs/profiles")
    parser.add_argument("--hardware", help="profil matériel (défaut : celui de cette machine)")
    parser.add_argument("--hardware-dir", default="configs/hardware")
    parser.add_argument("--models", default="configs/models.yaml", help="liste des modèles")
    parser.add_argument("--store", help="magasin SQLite (défaut : data/<profil>.db)")
    parser.add_argument(
        "--no-opensearch", action="store_true", help="recherche par le sens seulement"
    )
    args = parser.parse_args(argv)

    profile = load_profile(Path(args.profiles_dir) / args.profile).profile
    hardware = _hardware(args.hardware, Path(args.hardware_dir))
    spec = load_manifest(args.models).get(profile.embedding.model)
    if embedder_factory is None:
        from luciole.index.local_embedder import LocalEmbedder

        embedder_factory = LocalEmbedder
    embedder = embedder_factory(spec, hardware)
    chunker = Chunker(profile.chunking, load_tokenizer(profile.chunking.tokenizer))
    name = index_name(profile.name, embedder)
    vectors = QdrantIndex(qdrant or qdrant_client(), name, embedder.dimension)
    lexical = (
        None if args.no_opensearch else OpenSearchIndex(opensearch or opensearch_client(), name)
    )
    store_path = Path(args.store or Path("data") / f"{profile.name}.db")
    store_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Profil {profile.name}, machine {hardware.name} ({getattr(embedder, 'device', '?')}), "
        f"modèle {embedder.name}, index {name}, magasin {store_path}"
    )
    with MetadataStore(store_path, profile) as store:
        pipeline = IngestionPipeline(
            profile,
            store,
            default_registry(),
            chunker,
            embedder,
            vectors,
            lexical,
            batch_size=hardware.embedding_batch_size,
        )
        report = pipeline.run(args.root)
    print(report.summary())
    for skipped in report.ignored:
        print(f"  ignoré : {skipped.relative_path} ({skipped.reason})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
