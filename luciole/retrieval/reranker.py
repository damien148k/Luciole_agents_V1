"""Reranker : note chaque couple (question, passage) et trie.

Le vrai modèle est un cross-encoder ``sentence-transformers`` lu sur le disque
local à la révision épinglée de ``configs/models.yaml`` (rien n'est téléchargé
en fonctionnement). Périphérique et taille des lots viennent du profil
matériel. PyTorch n'est importé qu'à la construction : les tests unitaires
injectent un faux chargeur, ou un faux reranker qui respecte ``Reranker``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from luciole.core.hardware import HardwareProfile
from luciole.core.model_files import ModelSpec, require


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Un score par passage, plus haut = plus pertinent."""
        ...


def _load_cross_encoder(path: Path, device: str):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from sentence_transformers import CrossEncoder

    return CrossEncoder(str(path), device=device, local_files_only=True)


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


class LocalReranker:
    def __init__(
        self,
        spec: ModelSpec,
        hardware: HardwareProfile,
        *,
        root: Path | None = None,
        loader: Callable[[Path, str], object] = _load_cross_encoder,
        cuda_available: Callable[[], bool] = _cuda_available,
    ) -> None:
        path = require(spec, root)
        if hardware.device == "cuda" and not cuda_available():
            raise RuntimeError(
                f"le profil matériel « {hardware.name} » demande le GPU (cuda), mais "
                "PyTorch ne voit aucun GPU : choisir le profil « cpu » (--hardware cpu), "
                "ou passer par l'image Docker (GPU)."
            )
        self.spec = spec
        self.name = spec.short_name
        self.device = hardware.device
        self.batch_size = hardware.reranker_batch_size
        self._model = loader(path, hardware.device)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        scores = self._model.predict(
            [(query, p) for p in passages],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]
