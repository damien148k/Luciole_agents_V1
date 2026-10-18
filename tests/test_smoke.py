"""Test de fumée : chaque vérification rapporte vert ou un échec explicite,
sans PyTorch, modèle ni service (tout est remplacé par des doublures). Le
vrai passage se fait dans l'image Docker (``.\\DEV.ps1 -Action smoke``)."""

import sys
from pathlib import Path
from types import SimpleNamespace

from luciole.core.hardware import list_hardware_profiles
from luciole.core.model_files import Manifest, fetch
from luciole.smoke import (
    check_embedding,
    check_machine,
    check_models,
    check_services,
    check_torch,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from fakes import FakeEmbedder  # noqa: E402
from test_model_files import SPEC, fake_downloader  # noqa: E402

PROFILES = list_hardware_profiles(ROOT / "configs" / "hardware")


def fake_torch(cuda: bool):
    return lambda: SimpleNamespace(
        __version__="2.14.1",
        version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(
            is_available=lambda: cuda, get_device_name=lambda i: "NVIDIA GeForce RTX 3080 Ti"
        ),
    )


def test_machine_profil_suggere_ou_impose(monkeypatch):
    monkeypatch.setattr("luciole.smoke.detect_arch", lambda: "x86_64")
    check, profile = check_machine(PROFILES, None, 12)
    assert check.ok and profile.name == "rtx3080ti" and "suggéré" in check.detail
    check, profile = check_machine(PROFILES, "cpu", 12)
    assert check.ok and profile.name == "cpu" and "imposé" in check.detail
    check, profile = check_machine(PROFILES, "gx10", None)
    assert not check.ok and profile is None and "aarch64" in check.detail
    check, _ = check_machine(PROFILES, "inconnu", None)
    assert not check.ok and "aucun profil" in check.detail


def test_modeles(tmp_path):
    manifest = Manifest(models={"essai": SPEC})
    [missing] = check_models(manifest, tmp_path)
    assert not missing.ok and "dossier absent" in missing.detail
    fetch(SPEC, tmp_path, downloader=fake_downloader())
    [ready] = check_models(manifest, tmp_path)
    assert ready.ok and ready.detail == "org/modele@aaaaaaa"


def test_pytorch_et_gpu():
    gpu, cpu = PROFILES["rtx3080ti"], PROFILES["cpu"]
    ok = check_torch(gpu, fake_torch(True))
    assert ok.ok and "CUDA 13.0" in ok.detail and "3080 Ti" in ok.detail
    no_gpu = check_torch(gpu, fake_torch(False))
    assert not no_gpu.ok and "pilote NVIDIA 580" in no_gpu.detail
    assert check_torch(cpu, fake_torch(False)).ok

    def absent():
        raise ImportError("torch")

    assert "extra embed" in check_torch(cpu, absent).detail


def test_embedding():
    ok = check_embedding(lambda: FakeEmbedder())
    assert ok.ok and "dimension 64" in ok.detail

    def broken():
        raise FileNotFoundError("modèle BAAI/bge-m3@5617a9f pas prêt")

    failed = check_embedding(broken)
    assert not failed.ok and "pas prêt" in failed.detail


def test_services():
    class Up:
        def get_collections(self):
            return []

        def info(self):
            return {"version": {"number": "3.8.0"}}

    class Down:
        def get_collections(self):
            raise ConnectionError

        def info(self):
            raise ConnectionError

    assert all(c.ok for c in check_services(Up(), Up()))
    assert [c.ok for c in check_services(Down(), Up())] == [False, True]
    assert "3.8.0" in check_services(Up(), Up())[1].detail


def test_ligne_de_commande_rapporte_les_echecs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LUCIOLE_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("LUCIOLE_QDRANT_URL", ":memory:")
    monkeypatch.setenv("LUCIOLE_OPENSEARCH_URL", "http://127.0.0.1:1")
    code = main(
        [
            "--hardware",
            "cpu",
            "--hardware-dir",
            str(ROOT / "configs" / "hardware"),
            "--models",
            str(ROOT / "configs" / "models.yaml"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "[ÉCHEC] modèle bge-m3" in out and "[OK] qdrant" in out
    assert "[ÉCHEC] opensearch" in out and "échec(s)" in out
