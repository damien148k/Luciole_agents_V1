"""Profils matériels : une base de code, trois machines.

Tout réglage qui dépend de la machine (modèle LLM, fenêtre de contexte,
taille des lots, périphérique) vit dans ``configs/hardware/<machine>.yaml``,
jamais dans une constante. Le même code tourne sur :

- RTX 3080 Ti (12 Go, x86_64, Windows) : développement ;
- RTX A5000 (24 Go, x86_64) : production MRAe ;
- ASUS Ascent GX10 (GB10, 128 Go unifiés, **aarch64**) : production, gros modèles.

``python -m luciole.core.hardware`` décrit la machine courante et propose
un profil ; le test de fumée complet (modèles, PyTorch, embedding,
services) est ``python -m luciole.smoke``.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Arch = Literal["x86_64", "aarch64"]
Device = Literal["cuda", "cpu"]

# Normalisation des noms d'architecture renvoyés par platform.machine().
_ARCH_ALIASES: dict[str, Arch] = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
}


class HardwareProfile(BaseModel):
    """Contenu validé de ``configs/hardware/<machine>.yaml``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    title: str
    arch: Arch
    device: Device = "cuda"
    gpu_memory_gb: int = Field(ge=0)
    llm_model: str = Field(description="Tag Ollama, ex. qwen2.5:14b-instruct-q4_K_M")
    llm_num_ctx: int = Field(ge=2048)
    embedding_batch_size: int = Field(ge=1)
    reranker_batch_size: int = Field(ge=1)
    ocr_device: Device = "cuda"
    notes: str = ""


def load_hardware_profile(path: str | Path) -> HardwareProfile:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"profil matériel absent : {path}")
    with path.open(encoding="utf-8") as f:
        return HardwareProfile.model_validate(yaml.safe_load(f))


def list_hardware_profiles(root: str | Path) -> dict[str, HardwareProfile]:
    root = Path(root)
    if not root.is_dir():
        return {}
    return {p.stem: load_hardware_profile(p) for p in sorted(root.glob("*.yaml"))}


def detect_arch(machine: str | None = None) -> Arch:
    """Architecture du processeur, normalisée. Lève si inconnue plutôt que de deviner."""
    raw = (machine or platform.machine()).lower()
    if raw not in _ARCH_ALIASES:
        raise RuntimeError(f"architecture non prise en charge : {raw}")
    return _ARCH_ALIASES[raw]


def detect_gpu_memory_gb() -> int | None:
    """Mémoire du premier GPU NVIDIA via nvidia-smi ; None si indisponible."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    first = out.strip().splitlines()[0] if out.strip() else ""
    return round(int(first) / 1024) if first.isdigit() else None


def suggest_profile(
    profiles: dict[str, HardwareProfile], arch: Arch, gpu_memory_gb: int | None
) -> str | None:
    """Profil le plus proche de la machine : même architecture, puis mémoire GPU
    la plus grande qui ne dépasse pas celle détectée. Sans GPU : le profil CPU
    de cette architecture s'il existe."""
    candidates = [(n, p) for n, p in profiles.items() if p.arch == arch]
    if gpu_memory_gb is None:
        cpu = [n for n, p in candidates if p.device == "cpu"]
        return cpu[0] if cpu else None
    fitting = [
        (n, p) for n, p in candidates if p.device == "cuda" and p.gpu_memory_gb <= gpu_memory_gb
    ]
    if not fitting:
        return None
    return max(fitting, key=lambda np: np[1].gpu_memory_gb)[0]


def smoke(root: str | Path = "configs/hardware") -> int:
    """Décrit la machine et le profil suggéré. Code de retour 0 si un profil convient."""
    arch = detect_arch()
    gpu = detect_gpu_memory_gb()
    profiles = list_hardware_profiles(root)
    chosen = suggest_profile(profiles, arch, gpu)
    print(f"Système        : {platform.system()} {platform.release()}")
    print(f"Architecture   : {arch}")
    print(f"Python         : {sys.version.split()[0]}")
    print(f"GPU NVIDIA     : {'aucun détecté' if gpu is None else f'{gpu} Go'}")
    print(f"Profils connus : {', '.join(profiles) or 'aucun'}")
    if chosen is None:
        print("Profil suggéré : AUCUN — ajouter un profil dans configs/hardware/")
        return 1
    p = profiles[chosen]
    print(f"Profil suggéré : {chosen} ({p.title}) — LLM {p.llm_model}, fenêtre {p.llm_num_ctx}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(smoke())
