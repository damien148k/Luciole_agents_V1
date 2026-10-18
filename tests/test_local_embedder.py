"""Embedder réel, avec un faux chargeur : on vérifie le contrat (dimension,
norme 1, lots du profil matériel) et les erreurs claires, sans PyTorch ni
modèle. Le vrai BGE-M3 est essayé dans ``test_bge_m3.py`` (``integration``)."""

import sys
from pathlib import Path

import pytest

from luciole.core.hardware import load_hardware_profile
from luciole.core.model_files import fetch
from luciole.index.local_embedder import LocalEmbedder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_model_files import SPEC, fake_downloader  # noqa: E402

CPU = load_hardware_profile(ROOT / "configs" / "hardware" / "cpu.yaml")
GPU = load_hardware_profile(ROOT / "configs" / "hardware" / "rtx3080ti.yaml")


class FakeModel:
    def __init__(self, path: Path, device: str) -> None:
        self.path, self.device = path, device
        self.batches: list[int] = []

    def get_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, batch_size, normalize_embeddings, convert_to_numpy, show_progress_bar):
        assert normalize_embeddings and convert_to_numpy and not show_progress_bar
        self.batches.append(batch_size)
        return [[float(len(t)), 1.0, 0.0] for t in texts]


@pytest.fixture()
def models(tmp_path):
    fetch(SPEC, tmp_path, downloader=fake_downloader())
    return tmp_path


def test_contrat_de_l_embedder(models):
    embedder = LocalEmbedder(SPEC, CPU, root=models, loader=FakeModel)
    assert embedder.name == "org/modele@aaaaaaa" and embedder.dimension == 3
    assert embedder._model.path == models / "org" / "modele"
    assert embedder._model.device == "cpu"
    vectors = embedder.embed(["abc", "de"])
    assert len(vectors) == 2 and all(abs(sum(x * x for x in v) - 1) < 1e-9 for v in vectors)
    assert embedder._model.batches == [CPU.embedding_batch_size]
    assert embedder.embed([]) == []


def test_modele_absent_erreur_claire(tmp_path):
    with pytest.raises(FileNotFoundError, match="DEV.ps1 -Action models"):
        LocalEmbedder(SPEC, CPU, root=tmp_path, loader=FakeModel)


def test_gpu_demande_mais_absent(models):
    with pytest.raises(RuntimeError, match="--hardware cpu"):
        LocalEmbedder(SPEC, GPU, root=models, loader=FakeModel, cuda_available=lambda: False)
    embedder = LocalEmbedder(SPEC, GPU, root=models, loader=FakeModel, cuda_available=lambda: True)
    assert embedder._model.device == "cuda" and embedder.batch_size == GPU.embedding_batch_size


def test_dimension_inconnue_refusee(models):
    class NoDim(FakeModel):
        def get_embedding_dimension(self):
            return None

    with pytest.raises(ValueError, match="dimension inconnue"):
        LocalEmbedder(SPEC, CPU, root=models, loader=NoDim)
