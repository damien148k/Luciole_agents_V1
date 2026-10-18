"""Profils matériels : les quatre profils livrés sont valides et la
suggestion choisit le bon selon l'architecture et la mémoire GPU."""

from pathlib import Path

import pytest

from luciole.core.hardware import (
    detect_arch,
    list_hardware_profiles,
    load_hardware_profile,
    smoke,
    suggest_profile,
)

HARDWARE = Path(__file__).resolve().parents[1] / "configs" / "hardware"


@pytest.fixture(scope="module")
def profiles():
    return list_hardware_profiles(HARDWARE)


def test_les_profils_livres(profiles):
    assert set(profiles) == {"rtx3080ti", "a5000", "gx10", "cpu"}
    assert profiles["gx10"].arch == "aarch64"
    assert profiles["rtx3080ti"].llm_num_ctx < profiles["a5000"].llm_num_ctx


@pytest.mark.parametrize(
    "machine, attendu",
    [("x86_64", "x86_64"), ("AMD64", "x86_64"), ("aarch64", "aarch64"), ("arm64", "aarch64")],
)
def test_detect_arch(machine, attendu):
    assert detect_arch(machine) == attendu


def test_detect_arch_inconnue():
    with pytest.raises(RuntimeError):
        detect_arch("riscv64")


@pytest.mark.parametrize(
    "arch, gpu, attendu",
    [
        ("x86_64", 12, "rtx3080ti"),
        ("x86_64", 24, "a5000"),
        ("x86_64", 48, "a5000"),  # plus grosse carte x86 : le plus grand profil qui tient
        ("x86_64", 8, None),  # trop petit pour tout profil CUDA x86
        ("x86_64", None, "cpu"),
        ("aarch64", 96, "gx10"),
        ("aarch64", None, None),  # pas de profil CPU ARM pour l'instant
    ],
)
def test_suggest_profile(profiles, arch, gpu, attendu):
    assert suggest_profile(profiles, arch, gpu) == attendu


def test_smoke_tourne_sans_gpu(capsys):
    code = smoke(HARDWARE)
    out = capsys.readouterr().out
    assert "Architecture" in out and "Profil suggéré" in out
    assert code in (0, 1)


def test_gpu_invisible_pour_pytorch(monkeypatch):
    """Windows hors Docker : PyTorch sans CUDA, Ollama sur le GPU."""
    from luciole.core import hardware
    from luciole.index.pipeline import _hardware

    monkeypatch.setattr(hardware, "detect_arch", lambda: "x86_64")
    monkeypatch.setattr(hardware, "detect_gpu_memory_gb", lambda: 12)
    chosen = _hardware(None, HARDWARE, cuda_available=lambda: False)
    cpu = load_hardware_profile(HARDWARE / "cpu.yaml")
    gpu = load_hardware_profile(HARDWARE / "rtx3080ti.yaml")
    assert chosen.name == "rtx3080ti-cpu" and chosen.device == "cpu"
    assert (chosen.llm_model, chosen.llm_num_ctx) == (gpu.llm_model, gpu.llm_num_ctx)
    assert chosen.embedding_batch_size == cpu.embedding_batch_size
    assert chosen.ocr_device == "cpu"
    assert _hardware(None, HARDWARE, cuda_available=lambda: True) == gpu
    # Profil demandé : jamais changé.
    assert _hardware("rtx3080ti", HARDWARE, cuda_available=lambda: False) == gpu
