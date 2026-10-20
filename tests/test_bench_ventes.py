"""Banc « ventes » : les 10 dossiers fictifs d'``evaluation/ventes`` se lisent
sans erreur avec le profil, et le jeu de référence ne renvoie qu'à des
réunions qui existent."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from luciole.core.profile import load_profile
from luciole.evaluation.bench import cites_part, named_parts
from luciole.evaluation.reference import read_reference
from luciole.evaluation.spec import load_bench_spec
from luciole.ingestion.dossiers import scan
from luciole.ingestion.registry import default_registry

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "evaluation" / "ventes"
VENTES = load_profile(ROOT / "configs" / "profiles" / "ventes").profile
SPEC = load_bench_spec(ROOT / "configs" / "profiles" / "ventes")


def test_dix_dossiers_lisibles():
    result = scan(BENCH / "dossiers", VENTES, default_registry())
    assert len(result.dossiers) == 10 and not result.skipped
    assert all(d.valid for d in result.dossiers) and all(d.valid for d in result.documents)
    issues = Counter(d.metadata["issue"] for d in result.dossiers)
    assert issues == {"gagne": 4, "perdu": 4, "en_cours": 1, "abandonne": 1}
    assert {d.metadata["commercial"] for d in result.dossiers} == {
        "Camille Martin",
        "Julien Bernard",
        "Sofia Garnier",
    }
    assert all(d.metadata["client"].endswith("(fictif)") for d in result.dossiers)


def test_reference_renvoie_a_des_reunions_existantes():
    result = scan(BENCH / "dossiers", VENTES, default_registry())
    meetings = {
        d.dossier_id: {
            str(doc.metadata["numero_reunion"]) for doc in result.documents_of(d.dossier_id)
        }
        for d in result.dossiers
    }
    items = read_reference(BENCH / "reference.jsonl")
    assert len(items) == 10 and {i.dossier for i in items} == set(meetings)
    for item in items:
        assert item.expected, item.id
        assert {e.part for e in item.expected} <= meetings[item.dossier], item.id


def test_reunion_citee_par_son_nom_pas_par_sa_page():
    """Toutes les réunions font une page : « p. 1 » ne désigne aucune réunion."""
    [item] = [i for i in read_reference(BENCH / "reference.jsonl") if i.id == "syndic-mercier"]
    good = SimpleNamespace(
        answer="Signé à la démonstration [Source: syndic-mercier/reunion-2.transcript.md, p. 1]."
    )
    pages_only = SimpleNamespace(
        answer="Signé à la démonstration (p. 1).",
        citations=lambda: [{"cited": True}, {"cited": True}],
    )
    assert named_parts(good.answer, SPEC.reference) == ["2"]
    assert named_parts("Réunion 1 puis réunion 3", SPEC.reference) == ["1", "3"]
    assert cites_part(good, item, SPEC.reference, [], SPEC.metrics.cite_by_pages)
    assert not cites_part(pages_only, item, SPEC.reference, ["1", "2"], False)
    assert cites_part(pages_only, item, SPEC.reference, ["1", "2"], True)  # réglage MRAe
    assert SPEC.metrics.part_label == "réunion"


def test_banc_ventes_de_bout_en_bout(tmp_path):
    """Les 10 dossiers indexés en mémoire, le vrai bench.yaml et le vrai jeu de
    référence ; le LLM scripté cite toujours la réunion 2."""
    import json
    import sys

    sys.path.insert(0, str(ROOT / "tests"))
    from test_answer import ScaledReranker, ScriptedLLM
    from test_index_pipeline import Setup

    from luciole.evaluation.bench import main

    s = Setup(tmp_path, corpus=BENCH / "dossiers")
    s.pipeline.run(s.root)
    llm = ScriptedLLM("Le client a décidé à la démonstration (reunion-2.transcript.md).")
    lines: list[str] = []
    try:
        code = main(
            [
                "--profile",
                "ventes",
                "--hardware",
                "cpu",
                "--store",
                str(s.store.path),
                "--reference",
                str(BENCH / "reference.jsonl"),
                "--report",
                str(tmp_path / "rapport.json"),
                "--answer",
                "--runs",
                "1",
            ],
            log=lines.append,
            embedder_factory=lambda spec, hw: s.embedder,
            reranker_factory=lambda spec, hw: ScaledReranker(),
            qdrant=s.qdrant,
            opensearch=s.opensearch,
            llm_factory=lambda hw, gen: llm,
        )
    finally:
        s.store.close()
    assert code == 0
    rep = json.loads((tmp_path / "rapport.json").read_text("utf-8"))
    summary = rep["runs"][0]["summary"]
    assert summary["remarques"] == 10 == summary["avec_renvoi"]
    # Seule la clinique attend la réunion 3 sans la réunion 2.
    assert summary["tome_cite"] == 9 and summary["tome_fourni"] == 10
    for item in rep["runs"][0]["items"]:
        assert all(c.startswith(item["id"] + "/") for c in item["citations"])
        assert item["named_parts"] == ["2"]
    out = "\n".join(lines)
    assert "réunion citée dans la réponse : 9/10" in out and "bonne réunion @1" in out


def test_reunion_d_un_dossier_compare_ne_compte_pas():
    """Une boucle cite aussi des dossiers comparés : seule la réunion du dossier
    analysé compte."""
    text = (
        "Signé à la réunion 1 (syndic-mercier/reunion-1.transcript.md), comme chez "
        "un autre client (geometres-blanc/reunion-2.transcript.md)."
    )
    assert named_parts(text, SPEC.reference, "syndic-mercier") == ["1"]
    assert named_parts(text, SPEC.reference) == ["1", "2"]
    mrae = load_bench_spec(ROOT / "configs" / "profiles" / "mrae").reference
    assert named_parts(
        "Dossier Brissy/02_EIE - Tome 4 - Milieu naturel", mrae, "Dossier Brissy"
    ) == ["4"]


def test_banc_ventes_par_la_boucle(tmp_path):
    """--agent sur les 10 dossiers : la boucle lit chaque dossier, compare, et
    sa réponse cite la réunion 2 du dossier analysé."""
    import json
    import sys

    sys.path.insert(0, str(ROOT / "tests"))
    from test_agent_ventes import FakeLLM
    from test_answer import ScaledReranker
    from test_index_pipeline import Setup

    from luciole.evaluation.bench import main

    s = Setup(tmp_path, corpus=BENCH / "dossiers")
    s.pipeline.run(s.root)
    llm = FakeLLM(text="Décidé à la deuxième réunion (reunion-2.transcript.md).")
    try:
        code = main(
            [
                "--profile",
                "ventes",
                "--hardware",
                "cpu",
                "--store",
                str(s.store.path),
                "--reference",
                str(BENCH / "reference.jsonl"),
                "--report",
                str(tmp_path / "rapport.json"),
                "--agent",
                "--runs",
                "1",
            ],
            log=lambda line: None,
            embedder_factory=lambda spec, hw: s.embedder,
            reranker_factory=lambda spec, hw: ScaledReranker(),
            qdrant=s.qdrant,
            opensearch=s.opensearch,
            llm_factory=lambda hw, gen: llm,
        )
    finally:
        s.store.close()
    assert code == 0
    rep = json.loads((tmp_path / "rapport.json").read_text("utf-8"))
    summary = rep["runs"][0]["summary"]
    assert rep["mode"] == "boucle"
    assert summary["tome_cite"] == 9 and summary["tome_fourni"] == 10
    assert summary["esquives"] == 0
    # 28 réunions extraites une à une, puis une rédaction par dossier.
    assert sum(c["schema"] for c in llm.calls) == 28 and len(llm.calls) == 38
