"""Ligne de commande de la boucle : ``python -m luciole.agents.run``."""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_agent_mrae import ANALYSIS_JSON  # noqa: E402
from test_answer import EVASIVE, GOOD, QUESTION, ScaledReranker, ScriptedLLM  # noqa: E402
from test_index_pipeline import Setup  # noqa: E402

from luciole.agents.run import main  # noqa: E402


def test_ligne_de_commande(tmp_path, capsys, monkeypatch):
    s = Setup(tmp_path)
    s.pipeline.run(s.root)
    profiles = tmp_path / "profiles"
    shutil.copytree(ROOT / "configs" / "profiles" / "ventes", profiles / "ventes")
    for name in ("loop.yaml", "prompts.yaml"):  # la boucle MRAe et ses prompts
        shutil.copy(ROOT / "configs" / "profiles" / "mrae" / name, profiles / "ventes")
    monkeypatch.setenv("LUCIOLE_DATA_DIR", str(tmp_path / "data"))
    llm = ScriptedLLM(ANALYSIS_JSON, EVASIVE, GOOD)
    try:
        code = main(
            [
                QUESTION,
                "--profile",
                "ventes",
                "--profiles-dir",
                str(profiles),
                "--hardware",
                "cpu",
                "--store",
                str(s.store.path),
            ],
            embedder_factory=lambda spec, hw: s.embedder,
            reranker_factory=lambda spec, hw: ScaledReranker(),
            qdrant=s.qdrant,
            opensearch=s.opensearch,
            llm_factory=lambda hw, gen: llm,
        )
    finally:
        s.store.close()
    out = capsys.readouterr().out
    assert code == 0 and GOOD in out and "Verdict : SUFFISANT" in out
    assert (
        "Chemin : analyser > composer > rechercher_questions > rediger > verdict > elargir"
        " > rediger > verdict > reponse"
    ) in out
    assert "verdict != 'SUFFISANT' and refinements_left > 0 -> elargir" in out
    assert "Arrêt : reponse (8 étape(s))" in out
    traces = list((tmp_path / "data").rglob("*.jsonl"))
    assert traces, "la boucle écrit sa trace"
