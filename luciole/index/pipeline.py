"""Pipeline d'ingestion : une racine de dossiers → magasin + index alignés.

Un passage (``run``) :

1. parcourt la racine (``scan``) ; un document déjà indexé avec la même clé
   n'est **pas réanalysé** (sa fiche est relue dans le magasin) ;
2. pour chaque document **nouveau, modifié, ou indexé avec une autre clé**
   (autre embedder, autre schéma d'index, autre version du parser, profil
   de cas d'usage modifié) : découpe, calcule les vecteurs des
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
import hashlib
import json
import os
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
from luciole.core.versions import PARSER_VERSION
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


def document_key(profile: UseCaseProfile, embedder: Embedder) -> str:
    """Clé notée pour chaque document indexé : la clé d'index, la version du
    parser et l'empreinte de ce qui, dans le profil, change les fiches ou les
    morceaux (entités, règles d'ingestion, découpage, modèle). Changer une
    taille de morceau réindexe donc tout, sans changer de collection."""
    relevant = profile.model_dump(
        mode="json", include={"entities", "ingestion", "chunking", "embedding"}
    )
    digest = hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:8]
    return f"{index_key(embedder)}/parser{PARSER_VERSION}/profile-{digest}"


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
        self.key = document_key(profile, embedder)

    # ------------------------------------------------------------------
    def run(self, root: str | Path) -> PipelineReport:
        start = time.perf_counter()
        self.vectors.ensure()
        if self.lexical is not None:
            self.lexical.ensure()
        pending: dict[str, list[Chunk]] = {}
        skipped: list[str] = []

        def reuse(document_id: str) -> DocumentRecord | None:
            if self.store.indexed_with(document_id) != self.key:
                return None
            skipped.append(document_id)
            return self.store.get_document(document_id)

        def on_document(record: DocumentRecord, parsed: ParsedDocument) -> None:
            pending[record.document_id] = self.index_document(record, parsed)

        result = scan(root, self.profile, self.registry, on_document=on_document, reuse=reuse)
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
def _torch_sees_gpu() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _hardware(name: str | None, root: Path, cuda_available=_torch_sees_gpu):
    """Profil donné par ``--hardware``, sinon celui de la machine. Choisi
    automatiquement, un profil à GPU dont PyTorch ne voit pas le GPU garde son
    LLM et met les modèles sur le processeur (``models_on_cpu``)."""
    from luciole.core.hardware import (
        detect_arch,
        detect_gpu_memory_gb,
        list_hardware_profiles,
        models_on_cpu,
        suggest_profile,
    )

    profiles = list_hardware_profiles(root)
    if name is not None:
        if name not in profiles:
            raise SystemExit(f"profil matériel « {name} » inconnu ({', '.join(profiles)})")
        return profiles[name]
    name = suggest_profile(profiles, detect_arch(), detect_gpu_memory_gb())
    if name is None:
        raise SystemExit("aucun profil matériel ne convient : donner --hardware")
    chosen = profiles[name]
    if chosen.device == "cuda" and not cuda_available():
        chosen = models_on_cpu(chosen, profiles)
        logger.info(
            f"PyTorch ne voit pas le GPU : profil {chosen.name} (modèles sur le processeur, "
            f"LLM {chosen.llm_model})"
        )
    return chosen


def parser(prog: str, description: str) -> argparse.ArgumentParser:
    """Options communes au pipeline et au watcher."""
    cli = argparse.ArgumentParser(prog=prog, description=description)
    cli.add_argument("root", help="dossier racine (un sous-répertoire par dossier)")
    cli.add_argument("--profile", required=True, help="nom du cas d'usage (mrae, ventes…)")
    cli.add_argument("--profiles-dir", default="configs/profiles")
    cli.add_argument("--hardware", help="profil matériel (défaut : celui de cette machine)")
    cli.add_argument("--hardware-dir", default="configs/hardware")
    cli.add_argument("--models", default="configs/models.yaml", help="liste des modèles")
    cli.add_argument(
        "--store", help="magasin SQLite (défaut : $LUCIOLE_DATA_DIR ou data/, puis <profil>.db)"
    )
    cli.add_argument("--no-opensearch", action="store_true", help="recherche par le sens seulement")
    return cli


@dataclass
class Context:
    profile: UseCaseProfile
    store_path: Path
    store: MetadataStore
    pipeline: IngestionPipeline

    def __enter__(self) -> Context:
        return self

    def __exit__(self, *exc: object) -> None:
        self.store.close()


def build(
    args: argparse.Namespace,
    *,
    embedder_factory: Callable[..., Embedder] | None = None,
    qdrant=None,
    opensearch=None,
) -> Context:
    """Profil, modèle, index et magasin à partir des options.
    ``embedder_factory(spec, hardware)`` et les clients remplacent le vrai
    modèle et les vrais services dans les tests."""
    from luciole.core.model_files import load_manifest
    from luciole.core.profile import load_profile
    from luciole.index.clients import opensearch_client, qdrant_client
    from luciole.index.embedder import index_name
    from luciole.ingestion.registry import default_registry
    from luciole.ingestion.tokenizer import load_tokenizer

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
    data_dir = Path(os.environ.get("LUCIOLE_DATA_DIR") or "data")
    store_path = Path(args.store or data_dir / f"{profile.name}.db")
    store_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Profil {profile.name}, machine {hardware.name} ({getattr(embedder, 'device', '?')}), "
        f"modèle {embedder.name}, index {name}, magasin {store_path}"
    )
    store = MetadataStore(store_path, profile)
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
    return Context(profile, store_path, store, pipeline)


def main(argv: list[str] | None = None, **services) -> int:
    cli = parser(
        "python -m luciole.index.pipeline",
        "Indexe une racine de dossiers (Qdrant + OpenSearch) ; "
        "les fichiers inchangés ne sont pas recalculés.",
    )
    args = cli.parse_args(argv)
    with build(args, **services) as ctx:
        report = ctx.pipeline.run(args.root)
    print(report.summary())
    for skipped in report.ignored:
        print(f"  ignoré : {skipped.relative_path} ({skipped.reason})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
