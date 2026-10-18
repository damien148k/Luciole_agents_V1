"""Ligne de commande : poser une question à l'index d'un cas d'usage.

    python -m luciole.retrieval.search "question" --profile mrae
    python -m luciole.retrieval.search "question" --profile ventes --where dossier.issue=perdu

Affiche les passages retenus (citation, scores, début du texte). Les modèles
(embedding, reranker) sont lus sur le disque local ; rien n'est téléchargé.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

from luciole.retrieval.hybrid import HybridRetriever, RetrievalResult
from luciole.store.sqlite import Condition


def _condition(text: str) -> Condition:
    field, sep, value = text.partition("=")
    if not sep or not field.strip():
        raise argparse.ArgumentTypeError(f"filtre « {text} » : écrire champ=valeur")
    return Condition(field.strip(), "eq", value.strip())


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        prog="python -m luciole.retrieval.search",
        description="Recherche hybride (sens + mots), fusion RRF et reranker.",
    )
    cli.add_argument("question")
    add_service_options(cli)
    cli.add_argument(
        "--where", action="append", type=_condition, default=[], help="filtre champ=valeur"
    )
    cli.add_argument("--top-k", type=int, help="nombre de passages (défaut : profil)")
    return cli


def add_service_options(cli: argparse.ArgumentParser) -> None:
    """Cas d'usage, matériel, modèles, magasin et services : options communes à
    la recherche, à la réponse et au banc d'essai."""
    cli.add_argument("--profile", required=True, help="nom du cas d'usage (mrae, ventes…)")
    cli.add_argument("--profiles-dir", default="configs/profiles")
    cli.add_argument("--hardware", help="profil matériel (défaut : celui de cette machine)")
    cli.add_argument("--hardware-dir", default="configs/hardware")
    cli.add_argument("--models", default="configs/models.yaml")
    cli.add_argument("--store", help="magasin SQLite (défaut : $LUCIOLE_DATA_DIR ou data/)")
    cli.add_argument("--no-opensearch", action="store_true", help="recherche par le sens seulement")


def build(
    args: argparse.Namespace,
    *,
    embedder_factory: Callable | None = None,
    reranker_factory: Callable | None = None,
    qdrant=None,
    opensearch=None,
):
    """Retriever et magasin à partir des options ; les fabriques et les clients
    remplacent les vrais modèles et services dans les tests."""
    from luciole.core.model_files import load_manifest
    from luciole.core.profile import load_profile
    from luciole.index.clients import opensearch_client, qdrant_client
    from luciole.index.embedder import index_name
    from luciole.index.opensearch import OpenSearchIndex
    from luciole.index.pipeline import _hardware
    from luciole.index.qdrant import QdrantIndex
    from luciole.store.sqlite import MetadataStore

    profile = load_profile(Path(args.profiles_dir) / args.profile).profile
    hardware = _hardware(args.hardware, Path(args.hardware_dir))
    manifest = load_manifest(args.models)
    if embedder_factory is None:
        from luciole.index.local_embedder import LocalEmbedder as embedder_factory
    embedder = embedder_factory(manifest.get(profile.embedding.model), hardware)
    reranker = None
    if profile.retrieval.rerank_model is not None:
        if reranker_factory is None:
            from luciole.retrieval.reranker import LocalReranker as reranker_factory
        reranker = reranker_factory(manifest.get(profile.retrieval.rerank_model), hardware)
    name = index_name(profile.name, embedder)
    vectors = QdrantIndex(qdrant or qdrant_client(), name, embedder.dimension)
    lexical = (
        None if args.no_opensearch else OpenSearchIndex(opensearch or opensearch_client(), name)
    )
    data_dir = Path(os.environ.get("LUCIOLE_DATA_DIR") or "data")
    store_path = Path(args.store or data_dir / f"{profile.name}.db")
    if not store_path.is_file():
        raise SystemExit(f"magasin absent : {store_path} (indexer d'abord : -Action index)")
    store = MetadataStore(store_path, profile)
    retriever = HybridRetriever(profile.retrieval, embedder, vectors, lexical, store, reranker)
    return retriever, store


def show(result: RetrievalResult) -> str:
    lists = ", ".join(f"{n} {c}" for n, c in result.lists.items())
    lines = [
        f"Question : {result.query}",
        f"Listes : {lists or 'aucune'} ; {result.fused} fusionnés, "
        f"{result.reranked} notés par {result.reranker or 'aucun reranker'}"
        + (f", {result.stale} écartés (index en retard)" if result.stale else ""),
    ]
    if result.allowed_documents is not None:
        lines.append(f"Filtre : {result.allowed_documents} document(s) permis")
    if not result.passages:
        lines.append("Aucun passage.")
    for i, p in enumerate(result.passages, start=1):
        rerank = "" if p.rerank is None else f", reranker {p.rerank:.3f}"
        extract = " ".join(p.text.split())[:300]
        lines.append(f"{i}. {p.citation} (RRF {p.rrf:.4f}{rerank})\n   {extract}")
    return "\n".join(lines)


def main(argv: list[str] | None = None, **services) -> int:
    args = parser().parse_args(argv)
    retriever, store = build(args, **services)
    try:
        result = retriever.retrieve(args.question, conditions=args.where, top_k=args.top_k)
    finally:
        store.close()
    print(show(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
