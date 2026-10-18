"""Profils matériels : les quatre profils livrés sont valides et la
suggestion choisit le bon selon l'architecture et la mémoire GPU."""

from pathlib import Path

import pytest

from luciole.core.hardware import (
    detect_arch,
    list_hardware_profiles,
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
