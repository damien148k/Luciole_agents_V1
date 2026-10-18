"""Porte de l'étape 2 avec le **vrai** tokenizer BGE-M3 (machines de Damien).

Le tokenizer est lu dans le cache local (Hugging Face ou
``LUCIOLE_MODELS_DIR``), jamais téléchargé. S'il est absent, le test échoue
avec la liste des emplacements essayés : c'est l'installation qu'il faut
compléter, pas le test qu'il faut sauter.
"""

import sys
from pathlib import Path

import pytest

from luciole.core.profile import ChunkingSpec, load_profile
from luciole.ingestion.chunking import Chunker
from luciole.ingestion.dossiers import scan
from luciole.ingestion.pdf import PdfParser
from luciole.ingestion.registry import default_registry
from luciole.ingestion.tokenizer import load_tokenizer

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests" / "corpus" / "pdf"))
from build_pdf_corpus import build_all  # noqa: E402
from test_chunking import _assert_gate, _record  # noqa: E402


@pytest.fixture(scope="module")
def bge_m3():
    return load_tokenizer("BAAI/bge-m3")


def test_tokenizer_bge_m3_local(bge_m3):
    assert bge_m3.model_max_length == 8192
    # Deux tokens spéciaux (<s>, </s>) même pour un texte vide.
    assert bge_m3.count("") == 2
    assert bge_m3.count("Réunion de suivi avec le client") > 2


@pytest.mark.parametrize("profile", ["mrae", "ventes"])
def test_aucun_morceau_ne_depasse_la_limite(bge_m3, tmp_path, profile):
    """Porte de l'étape 2 : corpus de réunions et PDF synthétiques, réglages
    du profil puis réglages très serrés (pour forcer les découpes)."""
    spec = load_profile(ROOT / "configs" / "profiles" / profile).profile
    tight = ChunkingSpec(size_tokens=64, overlap_pct=20, parent_size_tokens=128, max_tokens=160)
    pdfs = build_all(tmp_path / "pdf")
    for chunking in (spec.chunking, tight):
        chunker = Chunker(chunking, bge_m3)
        for name in ("simple.pdf", "a3_dense.pdf", "dense_table.pdf", "two_up_rotated.pdf"):
            parsed = PdfParser().parse(pdfs[name])
            _assert_gate(chunker.chunk(parsed, _record(name)), chunking, bge_m3)
        scan(
            ROOT / "tests" / "corpus" / "ventes",
            spec,
            default_registry(),
            on_document=lambda rec, parsed, c=chunker, s=chunking: _assert_gate(
                c.chunk(parsed, rec), s, bge_m3
            ),
        )
