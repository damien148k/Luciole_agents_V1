"""Embedder réel : un modèle ``sentence-transformers`` lu sur le disque local.

Pour BGE-M3 : vecteur dense de 1024 flottants (pooling CLS), normalisé. Le
modèle vient de ``LUCIOLE_MODELS_DIR`` à la révision épinglée de
``configs/models.yaml`` ; rien n'est téléchargé (le chargement se fait en
mode hors ligne). Le périphérique et la taille des lots viennent du profil
matériel, jamais d'une constante.

PyTorch et ``sentence-transformers`` sont dans l'extra ``embed`` : ils ne
sont importés qu'à la construction, et les tests unitaires injectent un
faux chargeur.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path

from luciole.core.hardware import HardwareProfile
from luciole.core.model_files import ModelSpec, require
from luciole.index.embedder import normalize


def _load_sentence_transformer(path: Path, device: str):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(path), device=device, local_files_only=True)


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


class LocalEmbedder:
    """Respecte le contrat ``Embedder`` (nom, dimension, embed)."""

    def __init__(
        self,
        spec: ModelSpec,
        hardware: HardwareProfile,
        *,
        root: Path | None = None,
        loader: Callable[[Path, str], object] = _load_sentence_transformer,
        cuda_available: Callable[[], bool] = _cuda_available,
    ) -> None:
        path = require(spec, root)
        if hardware.device == "cuda" and not cuda_available():
            raise RuntimeError(
                f"le profil matériel « {hardware.name} » demande le GPU (cuda), mais "
                "PyTorch ne voit aucun GPU. Sous Windows, le PyTorch installé par uv "
                "fonctionne sur le processeur seulement : choisir le profil « cpu » "
                "(--hardware cpu), ou passer par l'image Docker (GPU)."
            )
        self.spec = spec
        self.name = spec.short_name
        self.device = hardware.device
        self.batch_size = hardware.embedding_batch_size
        self._model = loader(path, hardware.device)
        dimension = self._model.get_embedding_dimension()
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"{self.name} : dimension inconnue ({dimension!r})")
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # Renormalisé côté Python : le contrat (norme 1) ne dépend pas du modèle.
        return [normalize([float(v) for v in row]) for row in vectors]
