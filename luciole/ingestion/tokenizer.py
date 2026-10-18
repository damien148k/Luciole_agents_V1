"""Compter les tokens comme le modèle d'embedding les comptera.

V5 mesurait ses morceaux en caractères (1024 caractères ≈ 250 tokens) : la
taille réelle vue par BGE-M3 était inconnue. Ici le découpeur ne connaît que
le contrat ``TokenCounter`` ; en production c'est le tokenizer du modèle
(``tokenizer.json``), lu **dans le cache local**, jamais téléchargé (règle
« hors ligne »). Les tests unitaires utilisent un compteur factice.

Où le fichier est cherché, dans l'ordre :

1. le chemin donné, s'il existe (un ``tokenizer.json`` ou son dossier) ;
2. ``$LUCIOLE_MODELS_DIR/<organisation>/<modèle>/`` ;
3. le cache Hugging Face (``$HF_HUB_CACHE``, ``$HF_HOME/hub`` ou
   ``~/.cache/huggingface/hub``), là où V5 et ``sentence-transformers`` ont
   déjà rangé BGE-M3.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    """Ce dont le découpeur a besoin : compter, et connaître la limite du modèle."""

    #: Nom lisible (trace, rapport).
    name: str
    #: Longueur maximale acceptée par le modèle, tokens spéciaux compris ;
    #: ``None`` si inconnue.
    model_max_length: int | None

    def count(self, text: str) -> int:
        """Nombre de tokens, tokens spéciaux compris (ce que l'embedder verra)."""
        ...


class HfTokenCounter:
    """Tokenizer Hugging Face (bibliothèque ``tokenizers``) lu depuis un fichier."""

    def __init__(self, path: Path, *, name: str | None = None) -> None:
        from tokenizers import Tokenizer

        self.path = path
        self.name = name or path.parent.name
        self._tok = Tokenizer.from_file(str(path))
        # Pas de troncature ni de remplissage : on veut la vraie longueur.
        self._tok.no_truncation()
        self._tok.no_padding()
        self.model_max_length = _model_max_length(path.parent)

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=True).ids)


def _model_max_length(folder: Path) -> int | None:
    config = folder / "tokenizer_config.json"
    if not config.is_file():
        return None
    try:
        value = json.loads(config.read_text(encoding="utf-8")).get("model_max_length")
    except (OSError, ValueError):
        return None
    # Hugging Face écrit 1e30 quand la limite n'est pas renseignée.
    return int(value) if isinstance(value, int | float) and 0 < value < 1_000_000 else None


def _hf_cache_dirs() -> list[Path]:
    dirs = []
    if os.environ.get("HF_HUB_CACHE"):
        dirs.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        dirs.append(Path(os.environ["HF_HOME"]) / "hub")
    dirs.append(Path.home() / ".cache" / "huggingface" / "hub")
    return dirs


def candidate_paths(name_or_path: str) -> list[Path]:
    """Emplacements essayés pour ``tokenizer.json``, dans l'ordre."""
    out: list[Path] = []
    given = Path(name_or_path).expanduser()
    out.append(given if given.suffix == ".json" else given / "tokenizer.json")
    if os.environ.get("LUCIOLE_MODELS_DIR"):
        out.append(Path(os.environ["LUCIOLE_MODELS_DIR"]) / name_or_path / "tokenizer.json")
    if "/" in name_or_path and not given.is_absolute():
        repo = "models--" + name_or_path.replace("/", "--")
        for cache in _hf_cache_dirs():
            snapshots = cache / repo / "snapshots"
            if snapshots.is_dir():
                # Le plus récent d'abord.
                found = sorted(snapshots.glob("*/tokenizer.json"), key=lambda p: p.stat().st_mtime)
                out.extend(reversed(found))
    return out


def load_tokenizer(name_or_path: str) -> HfTokenCounter:
    """Charge le tokenizer depuis le disque ; erreur claire s'il est absent."""
    tried = candidate_paths(name_or_path)
    for path in tried:
        if path.is_file():
            name = name_or_path if not Path(name_or_path).exists() else path.parent.name
            return HfTokenCounter(path, name=name)
    listed = "\n".join(f"  - {p}" for p in tried)
    raise FileNotFoundError(
        f"tokenizer « {name_or_path} » introuvable sur cette machine (rien n'est "
        f"téléchargé). Emplacements essayés :\n{listed}\n"
        "Copier le dossier du modèle (avec tokenizer.json) dans LUCIOLE_MODELS_DIR, "
        "ou donner le chemin du fichier."
    )
