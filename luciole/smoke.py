"""Test de fumée d'une machine : tout ce qu'il faut pour indexer est-il là ?

Avant toute campagne, sur chaque machine (3080 Ti, A5000, GX10) :

1. **machine** : architecture, GPU NVIDIA, profil matériel retenu ;
2. **modèles** : chaque modèle de ``configs/models.yaml`` complet et à la
   bonne révision dans ``LUCIOLE_MODELS_DIR`` ;
3. **PyTorch** : installé, et voit le GPU si le profil le demande ;
4. **embedding** : le modèle du profil calcule un vecteur sur ce périphérique ;
5. **services** : Qdrant et OpenSearch répondent.

Code de retour 0 seulement si tout est vert. Dans l'image Docker (GPU) :
``.\\DEV.ps1 -Action smoke`` ; sur la machine elle-même :
``uv run --extra embed python -m luciole.smoke --hardware cpu``.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from luciole.core.hardware import (
    HardwareProfile,
    detect_arch,
    detect_gpu_memory_gb,
    list_hardware_profiles,
    suggest_profile,
)
from luciole.core.model_files import Manifest, check, load_manifest, models_dir


@dataclass
class Check:
    name: str
    ok: bool
    detail: str

    def line(self) -> str:
        return f"  [{'OK' if self.ok else 'ÉCHEC'}] {self.name:<10} {self.detail}"


def check_machine(
    profiles: dict[str, HardwareProfile], forced: str | None, gpu_memory_gb: int | None
) -> tuple[Check, HardwareProfile | None]:
    arch = detect_arch()
    gpu = "aucun GPU NVIDIA" if gpu_memory_gb is None else f"GPU {gpu_memory_gb} Go"
    name = forced or suggest_profile(profiles, arch, gpu_memory_gb)
    if name is None or name not in profiles:
        known = ", ".join(profiles) or "aucun"
        return Check("machine", False, f"{arch}, {gpu} : aucun profil ne convient ({known})"), None
    profile = profiles[name]
    if profile.arch != arch:
        return Check(
            "machine", False, f"profil {name} prévu pour {profile.arch}, machine {arch}"
        ), None
    how = "imposé" if forced else "suggéré"
    return Check(
        "machine", True, f"{platform.system()} {arch}, {gpu}, profil {name} ({how})"
    ), profile


def check_models(manifest: Manifest, root: Path | None = None) -> list[Check]:
    out = []
    for name, spec in manifest.models.items():
        problems = check(spec, root)
        detail = spec.short_name if not problems else f"{spec.short_name} : {'; '.join(problems)}"
        out.append(Check(f"modèle {name}", not problems, detail))
    return out


def check_torch(
    profile: HardwareProfile, import_torch: Callable[[], object] | None = None
) -> Check:
    try:
        torch = (import_torch or _import_torch)()
    except ImportError:
        return Check(
            "pytorch", False, "absent : installer l'extra embed (.\\DEV.ps1 -Action install)"
        )
    version = torch.__version__
    if not torch.cuda.is_available():
        if profile.device == "cuda":
            return Check(
                "pytorch",
                False,
                f"{version} ne voit aucun GPU alors que le profil {profile.name} le demande "
                "(pilote NVIDIA 580 ou plus pour CUDA 13 ; "
                f"sous Windows, passer par l'image Docker){_cuda_cause(torch)}",
            )
        return Check("pytorch", True, f"{version}, processeur (profil {profile.name})")
    gpu = torch.cuda.get_device_name(0)
    return Check("pytorch", True, f"{version}, CUDA {torch.version.cuda}, {gpu}")


def _cuda_cause(torch) -> str:
    """Raison donnée par PyTorch lui-même (pilote trop ancien, aucun GPU...)."""
    try:
        torch.cuda.init()
    except Exception as e:  # noqa: BLE001 - on ne fait que rapporter le message
        return f" ; PyTorch dit : {' '.join(str(e).split())}"
    return ""


def _import_torch():
    import torch

    return torch


def check_embedding(make_embedder: Callable[[], object]) -> Check:
    try:
        start = time.perf_counter()
        embedder = make_embedder()
        loaded = time.perf_counter() - start
        start = time.perf_counter()
        vectors = embedder.embed(["Étude d'impact d'un parc éolien.", "Réunion de suivi client."])
        took = time.perf_counter() - start
    except Exception as e:  # noqa: BLE001 - le test de fumée rapporte, il ne s'arrête pas
        return Check("embedding", False, f"{type(e).__name__} : {' '.join(str(e).split())}")
    if len(vectors) != 2 or any(len(v) != embedder.dimension for v in vectors):
        return Check("embedding", False, f"{embedder.name} : vecteurs inattendus")
    device = getattr(embedder, "device", "?")
    return Check(
        "embedding",
        True,
        f"{embedder.name} sur {device}, dimension {embedder.dimension}, "
        f"chargé en {loaded:.1f} s, 2 textes en {took:.2f} s",
    )


def check_services(qdrant, opensearch) -> list[Check]:
    out = []
    try:
        qdrant.get_collections()
        out.append(Check("qdrant", True, "répond"))
    except Exception as e:  # noqa: BLE001
        out.append(Check("qdrant", False, f"injoignable ({type(e).__name__})"))
    try:
        version = opensearch.info()["version"]["number"]
        out.append(Check("opensearch", True, f"répond (version {version})"))
    except Exception as e:  # noqa: BLE001
        out.append(Check("opensearch", False, f"injoignable ({type(e).__name__})"))
    return out


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(
        prog="python -m luciole.smoke", description=__doc__.split("\n")[0]
    )
    cli.add_argument("--hardware", help="profil matériel (défaut : suggéré pour cette machine)")
    cli.add_argument("--hardware-dir", default="configs/hardware")
    cli.add_argument("--models", default="configs/models.yaml")
    cli.add_argument(
        "--embedding-model", default="bge-m3", help="modèle essayé (configs/models.yaml)"
    )
    args = cli.parse_args(argv)

    print(f"Test de fumée (Python {sys.version.split()[0]}, modèles dans {models_dir()})")
    checks: list[Check] = []
    machine, profile = check_machine(
        list_hardware_profiles(args.hardware_dir), args.hardware, detect_gpu_memory_gb()
    )
    checks.append(machine)
    manifest = load_manifest(args.models)
    checks += check_models(manifest)
    if profile is not None:
        checks.append(check_torch(profile))
        spec = manifest.get(args.embedding_model)

        def make():
            from luciole.index.local_embedder import LocalEmbedder

            return LocalEmbedder(spec, profile)

        checks.append(check_embedding(make))
    from luciole.index.clients import opensearch_client, qdrant_client

    checks += check_services(qdrant_client(), opensearch_client())
    for c in checks:
        print(c.line())
    failed = [c for c in checks if not c.ok]
    print("Résultat : " + ("tout est vert" if not failed else f"{len(failed)} échec(s)"))
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
