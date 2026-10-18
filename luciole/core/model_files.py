"""Fichiers des modèles locaux : où ils sont, à quelle révision, complets ou non.

En production, aucun modèle n'est téléchargé (règle « hors ligne ») : les
fichiers sont récupérés **une fois**, pendant la préparation de la machine,
à la **révision épinglée** dans ``configs/models.yaml``, puis lus depuis
``LUCIOLE_MODELS_DIR/<organisation>/<modèle>/`` (par défaut ``models/`` sous
le dossier courant, ignoré par git).

Après une récupération, un fichier ``luciole-model.json`` note le dépôt, la
révision et la taille de chaque fichier. ``check`` le relit : un modèle
incomplet, d'une autre révision ou modifié est signalé, jamais deviné.

    python -m luciole.core.model_files fetch bge-m3   # en ligne, une fois
    python -m luciole.core.model_files check          # hors ligne
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MANIFEST = Path("configs") / "models.yaml"
MARKER = "luciole-model.json"


class ModelSpec(BaseModel):
    """Un modèle de ``configs/models.yaml``."""

    model_config = ConfigDict(extra="forbid")

    repo: str = Field(pattern=r"^[\w.-]+/[\w.-]+$", description="dépôt Hugging Face")
    #: Empreinte complète du commit : une branche (« main ») bougerait.
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    role: str = Field(description="embedding, reranker…")
    #: Fichiers à récupérer, chemins relatifs dans le dépôt ; tous obligatoires.
    files: list[str] = Field(min_length=1)

    @property
    def short_name(self) -> str:
        """Nom lisible, révision comprise (« BAAI/bge-m3@5617a9f »)."""
        return f"{self.repo}@{self.revision[:7]}"


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: dict[str, ModelSpec]

    def get(self, name: str) -> ModelSpec:
        if name not in self.models:
            raise KeyError(
                f"modèle « {name} » absent de configs/models.yaml "
                f"(connus : {', '.join(self.models)})"
            )
        return self.models[name]


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> Manifest:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"liste des modèles absente : {path}")
    with path.open(encoding="utf-8") as f:
        return Manifest.model_validate(yaml.safe_load(f))


def models_dir() -> Path:
    """Racine des modèles locaux : ``$LUCIOLE_MODELS_DIR`` ou ``models/``."""
    return Path(os.environ.get("LUCIOLE_MODELS_DIR") or "models").expanduser()


def model_path(spec: ModelSpec, root: Path | None = None) -> Path:
    return (root or models_dir()) / spec.repo


def check(spec: ModelSpec, root: Path | None = None) -> list[str]:
    """Problèmes du modèle sur disque ; liste vide s'il est complet et à la
    bonne révision. Ne lit que les tailles (rapide, même pour 2 Go)."""
    folder = model_path(spec, root)
    if not folder.is_dir():
        return [f"dossier absent : {folder}"]
    marker = folder / MARKER
    if not marker.is_file():
        return [f"{MARKER} absent dans {folder} : modèle non récupéré par Luciole"]
    try:
        noted = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"{marker} illisible ({e})"]
    problems = []
    if noted.get("repo") != spec.repo or noted.get("revision") != spec.revision:
        problems.append(
            f"révision {noted.get('repo')}@{noted.get('revision')} sur disque, "
            f"{spec.repo}@{spec.revision} attendue"
        )
    sizes = noted.get("files", {})
    for name in spec.files:
        path = folder / name
        if not path.is_file():
            problems.append(f"fichier manquant : {name}")
        elif name in sizes and path.stat().st_size != sizes[name]:
            problems.append(f"fichier modifié depuis la récupération : {name}")
        elif name not in sizes:
            problems.append(f"fichier non noté à la récupération : {name}")
    return problems


def require(spec: ModelSpec, root: Path | None = None) -> Path:
    """Dossier du modèle, ou erreur claire s'il n'est pas prêt."""
    problems = check(spec, root)
    if problems:
        listed = "\n".join(f"  - {p}" for p in problems)
        raise FileNotFoundError(
            f"modèle {spec.short_name} pas prêt sur cette machine (rien n'est "
            f"téléchargé en fonctionnement) :\n{listed}\n"
            "Le récupérer une fois avec : .\\DEV.ps1 -Action models"
        )
    return model_path(spec, root)


def fetch(spec: ModelSpec, root: Path | None = None, downloader=None) -> Path:
    """Récupère les fichiers du modèle à sa révision épinglée (préparation de
    la machine, en ligne), puis note les tailles. ``downloader`` remplace
    ``huggingface_hub.snapshot_download`` dans les tests."""
    folder = model_path(spec, root)
    folder.mkdir(parents=True, exist_ok=True)
    if downloader is None:
        from huggingface_hub import snapshot_download as downloader
    downloader(
        repo_id=spec.repo,
        revision=spec.revision,
        local_dir=str(folder),
        allow_patterns=list(spec.files),
    )
    missing = [name for name in spec.files if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{spec.short_name} : absent(s) du dépôt à cette révision : {', '.join(missing)}"
        )
    noted = {
        "repo": spec.repo,
        "revision": spec.revision,
        "files": {name: (folder / name).stat().st_size for name in spec.files},
    }
    (folder / MARKER).write_text(json.dumps(noted, indent=2) + "\n", encoding="utf-8")
    return folder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m luciole.core.model_files",
        description="Récupérer (une fois, en ligne) ou vérifier (hors ligne) les modèles locaux.",
    )
    parser.add_argument("action", choices=["fetch", "check"])
    parser.add_argument("names", nargs="*", help="modèles (défaut : tous ceux de la liste)")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    names = args.names or list(manifest.models)
    print(f"Dossier des modèles : {models_dir().resolve()}")
    failed = 0
    for name in names:
        spec = manifest.get(name)
        if args.action == "fetch":
            print(f"Récupération de {spec.short_name} ({len(spec.files)} fichiers)…")
            try:
                fetch(spec)
            except Exception as e:  # noqa: BLE001 - réseau, révision inconnue, fichier absent
                failed += 1
                print(f"  [ÉCHEC] {name} : {e}")
                continue
        problems = check(spec)
        if problems:
            failed += 1
            print(f"  [ÉCHEC] {name} ({spec.short_name})")
            for p in problems:
                print(f"     - {p}")
        else:
            size = sum((model_path(spec) / f).stat().st_size for f in spec.files)
            print(f"  [OK] {name} ({spec.short_name}, {size / 1e9:.2f} Go)")
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
