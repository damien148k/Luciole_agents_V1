"""Réponse sourcée de bout en bout avec les **vrais** modèles (machines de
Damien) : BGE-M3, le reranker, Qdrant et OpenSearch, et le LLM du profil
matériel servi par Ollama (``LUCIOLE_OLLAMA_URL``, défaut 127.0.0.1:11434).

Le LLM est celui du profil matériel de la machine (GPU vu par nvidia-smi), même
quand PyTorch tourne sur le processeur : sous Windows hors Docker, PyTorch n'a
pas CUDA, alors qu'Ollama emploie le GPU lui-même.

Si Ollama ou son modèle manque, le test échoue en disant quoi installer :
c'est la machine qu'il faut compléter, pas le test qu'il faut sauter.
"""

import sys
from pathlib import Path

import httpx
import pytest

from luciole.core.hardware import (
    detect_arch,
    detect_gpu_memory_gb,
    list_hardware_profiles,
    suggest_profile,
)
from luciole.generation.answer import Answerer
from luciole.generation.llm import OllamaLLM, ollama_url
from luciole.generation.prompts import load_prompts

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_retrieval_real import VENTES, hardware, indexed, reranker  # noqa: E402,F401


@pytest.fixture(scope="module")
def machine(hardware):  # noqa: F811
    """Profil des modèles (PyTorch) avec le LLM et la fenêtre du profil de la machine."""
    profiles = list_hardware_profiles(ROOT / "configs" / "hardware")
    name = suggest_profile(profiles, detect_arch(), detect_gpu_memory_gb()) or hardware.name
    chosen = profiles[name]
    print(f"\nLLM du profil {name} : {chosen.llm_model}, fenêtre {chosen.llm_num_ctx}")
    return hardware.model_copy(
        update={"llm_model": chosen.llm_model, "llm_num_ctx": chosen.llm_num_ctx}
    )


@pytest.fixture(scope="module")
def llm(machine):
    try:
        tags = httpx.get(f"{ollama_url()}/api/tags", timeout=10).json()
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"Ollama injoignable sur {ollama_url()} ({e}) : lancer Ollama")
    names = {m.get("name") for m in tags.get("models", [])}
    if machine.llm_model not in names:
        pytest.fail(f"modèle absent d'Ollama : « ollama pull {machine.llm_model} »")
    return OllamaLLM(machine.llm_model, num_ctx=machine.llm_num_ctx)


QUESTION = "Pourquoi l'atelier Dubois a-t-il reporté son achat ?"


def check(result):
    print(f"\n{result.answer}\n\nVerdict {result.verdict}, confiance {result.confidence}")
    print(f"Gardes : {result.guards.to_dict()}")
    assert result.verdict in ("SUFFISANT", "PARTIEL") and not result.guards.evasive
    assert "atelier-dubois/reunion-3.transcript.md" in result.to_dict()["sources_used"]
    flat = result.answer.lower()
    assert any(w in flat for w in ("douze mille", "12 000", "12000", "budget", "embauche"))


def test_reponse_sourcee(indexed, llm, machine):  # noqa: F811
    """Deux appels de suite doivent tous deux passer les mêmes contrôles.

    Le texte n'est pas comparé au caractère près : le 2026-10-18 sur la 3080 Ti,
    même graine et température 0, le second appel a rendu une formulation
    différente (Ollama réutilise le début du prompt déjà calculé, et les calculs
    sur GPU ne sont pas identiques bit à bit). Le banc d'essai mesure ce bruit."""
    prompts = load_prompts(ROOT / "configs" / "profiles" / "ventes")
    answerer = Answerer(VENTES, prompts, indexed, llm, machine)
    check(answerer.answer(QUESTION))
    check(answerer.answer(QUESTION))
