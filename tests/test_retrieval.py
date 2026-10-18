"""Outil ``retrieve`` : fusion RRF, reranker, remontée au parent, filtres par
métadonnées, sur le corpus « ventes » indexé avec un faux embedder, Qdrant en
mémoire et un faux OpenSearch. Le vrai reranker est essayé dans
``test_retrieval_real.py`` (``integration``)."""

import re
import sys
from pathlib import Path

import pytest

from luciole.core.hardware import load_hardware_profile
from luciole.core.model_files import fetch, load_manifest
from luciole.core.profile import RetrievalSpec, load_profile
from luciole.retrieval.fusion import rrf
from luciole.retrieval.hybrid import HybridRetriever
from luciole.retrieval.lot import is_secondary, is_table_of_contents, strip_accents
from luciole.retrieval.reranker import LocalReranker
from luciole.retrieval.search import main as search_main
from luciole.store.sqlite import Condition

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_index_pipeline import Setup  # noqa: E402
from test_model_files import SPEC, fake_downloader  # noqa: E402

CPU = load_hardware_profile(ROOT / "configs" / "hardware" / "cpu.yaml")
GPU = load_hardware_profile(ROOT / "configs" / "hardware" / "rtx3080ti.yaml")
QUERY = "douze mille euros au-dessus de ce qu'on peut engager"
_WORD = re.compile(r"\w+")


class WordReranker:
    """Score = mots communs avec la question ; ``boost`` favorise un texte."""

    name = "faux-reranker"

    def __init__(self, boost: str | None = None) -> None:
        self.boost = boost
        self.calls: list[int] = []

    def score(self, query, passages):
        self.calls.append(len(passages))
        words = set(_WORD.findall(query.lower()))
        out = []
        for p in passages:
            s = float(len(words & set(_WORD.findall(p.lower()))))
            if self.boost and self.boost in p:
                s += 100.0
            out.append(s)
        return out


@pytest.fixture()
def env(tmp_path):
    s = Setup(tmp_path)
    s.pipeline.run(s.root)
    yield s
    s.store.close()


def retriever(env, spec: RetrievalSpec | None = None, reranker=None, lexical=True):
    spec = spec or RetrievalSpec(rerank_top_k=3, rerank_candidates=10)
    if reranker is None and spec.rerank_model is not None:
        reranker = WordReranker()
    return HybridRetriever(
        spec, env.embedder, env.vectors, env.lexical if lexical else None, env.store, reranker
    )


# --- fusion ---------------------------------------------------------------
def test_rrf_additionne_les_rangs():
    fused = rrf({"a": ["x", "y", "z"], "b": ["y", "w"]}, k=60)
    assert [f.chunk_id for f in fused] == ["y", "x", "w", "z"]
    y = fused[0]
    assert y.ranks == {"a": 2, "b": 1}
    assert y.score == pytest.approx(1 / 62 + 1 / 61)
    # Seuls les rangs comptent ; un doublon dans une liste compte une fois.
    assert rrf({"a": ["x", "x", "y"]}, k=1)[1].ranks == {"a": 3}
    # À score égal, ordre de première apparition.
    assert [f.chunk_id for f in rrf({"a": ["p"], "b": ["q"]}, k=60)] == ["p", "q"]
    assert rrf({}, k=60) == []
    with pytest.raises(ValueError):
        rrf({"a": ["x"]}, k=0)


# --- recherche hybride ----------------------------------------------------
def test_bon_passage_cite_avec_son_parent(env):
    result = retriever(env).retrieve(QUERY)
    assert set(result.lists) == {"dense:content", "dense:context", "bm25"}
    top = result.passages[0]
    assert top.unit.relative_path == "atelier-dubois/reunion-3.transcript.md"
    assert top.unit.kind == "parent" and top.hit.kind == "child"
    assert top.hit.parent_id == top.unit.chunk_id and top.hit.text in top.text
    assert top.citation.startswith("atelier-dubois/reunion-3.transcript.md")
    assert set(top.ranks) <= set(result.lists) and top.rerank is not None
    assert len(result.passages) <= 3 and result.reranker == "faux-reranker"
    trace = result.trace()
    assert trace["passages"][0]["citation"] == top.citation
    assert trace["passages"][0]["text"] is None  # la trace ne recopie pas les textes


def test_un_passage_par_parent_et_top_k(env):
    result = retriever(env, RetrievalSpec(rerank_top_k=50, rerank_candidates=50)).retrieve(QUERY)
    units = [p.unit.chunk_id for p in result.passages]
    assert len(units) == len(set(units))
    assert len(units) == env.store.count_chunks("parent")
    assert len(retriever(env).retrieve(QUERY, top_k=1).passages) == 1


def test_le_reranker_decide_l_ordre(env):
    target = env.store.select("document", [Condition("dossier_id", "eq", "cabinet-roche")])[0]
    marker = env.store.chunks_of(target.document_id, "child")[0].text[:40]
    reranker = WordReranker(boost=marker)
    spec = RetrievalSpec(rerank_top_k=3, rerank_candidates=50)
    result = retriever(env, spec, reranker=reranker).retrieve(QUERY)
    assert result.passages[0].unit.document_id == target.document_id
    assert reranker.calls == [result.reranked] and result.reranked == result.fused
    capped = WordReranker()
    retriever(env, reranker=capped).retrieve(QUERY)
    assert capped.calls == [10]  # rerank_candidates


def test_sans_reranker_ordre_rrf(env):
    spec = RetrievalSpec(rerank_model=None, rerank_top_k=3, rerank_candidates=3)
    result = retriever(env, spec).retrieve(QUERY)
    assert result.reranked == 0 and result.reranker is None
    assert all(p.rerank is None for p in result.passages)
    rrfs = [p.rrf for p in result.passages]
    assert rrfs == sorted(rrfs, reverse=True)
    with pytest.raises(ValueError, match="reranker"):
        HybridRetriever(RetrievalSpec(), env.embedder, env.vectors, None, env.store, None)


def test_sens_seulement(env):
    result = retriever(env, lexical=False).retrieve(QUERY)
    assert set(result.lists) == {"dense:content", "dense:context"} and result.passages


def test_filtre_par_metadonnees(env):
    r = retriever(env)
    won = r.retrieve(QUERY, conditions=[Condition("dossier.issue", "eq", "gagne")])
    assert won.passages and {p.unit.dossier_id for p in won.passages} == {"cabinet-roche"}
    assert won.allowed_documents == 2
    third = r.retrieve(QUERY, conditions=[Condition("numero_reunion", "eq", 3)])
    assert {p.unit.relative_path for p in third.passages} == {
        "atelier-dubois/reunion-3.transcript.md"
    }
    ids = [d.document_id for d in env.store.select("document")]
    both = r.retrieve(
        QUERY, document_ids=ids[:1], conditions=[Condition("numero_reunion", "ge", 1)]
    )
    assert {p.unit.document_id for p in both.passages} == {ids[0]}


def test_filtre_vide_n_elargit_pas(env):
    calls = env.embedder.calls
    result = retriever(env).retrieve(
        QUERY, conditions=[Condition("dossier.issue", "eq", "abandonne")]
    )
    assert result.passages == [] and result.allowed_documents == 0
    assert env.embedder.calls == calls  # aucune recherche lancée
    assert retriever(env).retrieve(QUERY, document_ids=[]).passages == []


def test_point_inconnu_du_magasin_ecarte(env):
    real_get = env.store.get_chunk
    first = retriever(env).retrieve(QUERY).passages[0].hit.chunk_id
    env.store.get_chunk = lambda cid: None if cid == first else real_get(cid)
    result = retriever(env).retrieve(QUERY)
    assert result.stale == 1 and first not in {p.hit.chunk_id for p in result.passages}


def test_question_vide(env):
    with pytest.raises(ValueError, match="vide"):
        retriever(env).retrieve("   ")


def test_reglages_du_profil():
    with pytest.raises(ValueError, match="rerank_candidates"):
        RetrievalSpec(rerank_top_k=10, rerank_candidates=5)
    with pytest.raises(ValueError):
        RetrievalSpec(dense_vectors=[])
    for name in ("mrae", "ventes"):
        spec = load_profile(ROOT / "configs" / "profiles" / name).profile.retrieval
        manifest = load_manifest(ROOT / "configs" / "models.yaml")
        assert manifest.get(spec.rerank_model).role == "reranker"


# --- reranker réel, faux chargeur -----------------------------------------
class FakeCrossEncoder:
    def __init__(self, path: Path, device: str) -> None:
        self.path, self.device = path, device
        self.batches: list[int] = []

    def predict(self, pairs, batch_size, show_progress_bar):
        assert not show_progress_bar
        self.batches.append(batch_size)
        return [float(len(p)) for _, p in pairs]


@pytest.fixture()
def models(tmp_path):
    fetch(SPEC, tmp_path, downloader=fake_downloader())
    return tmp_path


def test_reranker_local(models):
    r = LocalReranker(SPEC, CPU, root=models, loader=FakeCrossEncoder)
    assert r.name == "org/modele@aaaaaaa" and r._model.device == "cpu"
    assert r.score("q", ["abc", "a"]) == [3.0, 1.0]
    assert r._model.batches == [CPU.reranker_batch_size]
    assert r.score("q", []) == []


def test_reranker_erreurs_claires(models, tmp_path):
    with pytest.raises(RuntimeError, match="GPU"):
        LocalReranker(SPEC, GPU, root=models, loader=FakeCrossEncoder, cuda_available=lambda: False)
    with pytest.raises(FileNotFoundError, match="DEV.ps1 -Action models"):
        LocalReranker(SPEC, CPU, root=tmp_path / "vide", loader=FakeCrossEncoder)


# --- ligne de commande ------------------------------------------------------
def test_ligne_de_commande(env, capsys):
    code = search_main(
        [QUERY, "--profile", "ventes", "--hardware", "cpu", "--store", str(env.store.path)],
        embedder_factory=lambda spec, hw: env.embedder,
        reranker_factory=lambda spec, hw: WordReranker(),
        qdrant=env.qdrant,
        opensearch=env.opensearch,
    )
    out = capsys.readouterr().out
    assert (
        code == 0 and "1. atelier-dubois/reunion-3.transcript.md" in out and "faux-reranker" in out
    )
    search_main(
        [
            QUERY,
            "--profile",
            "ventes",
            "--hardware",
            "cpu",
            "--store",
            str(env.store.path),
            "--where",
            "dossier.issue=abandonne",
        ],
        embedder_factory=lambda spec, hw: env.embedder,
        reranker_factory=lambda spec, hw: WordReranker(),
        qdrant=env.qdrant,
        opensearch=env.opensearch,
    )
    out = capsys.readouterr().out
    assert "0 document(s) permis" in out and "Aucun passage." in out


# --- règles du lot (V5 v3.6) ----------------------------------------------
TOC = """Sommaire
3.1 Contexte réglementaire ........ 12
3.2 Milieu physique ........ 18
3.2.1 Géologie ........ 19
3.2.2 Hydrologie ........ 23
Carte 12 : Réseau hydrographique 24
Figure 4 – Coupe géologique (page 20)
"""


def test_sommaire_reconnu():
    assert is_table_of_contents(TOC)
    prose = "La MRAe recommande de compléter l'analyse.\n" * 6
    assert not is_table_of_contents(prose)
    assert not is_table_of_contents(
        "3.1 Contexte ........ 12\n3.2 Milieu ........ 18"
    )  # trop court
    mixed = TOC + "\n".join(f"Phrase de contenu numéro {i} sans page." for i in range(20))
    assert not is_table_of_contents(mixed)  # moins de 40 % de lignes de sommaire


def test_documents_secondaires():
    patterns = ["RNT", "resume_non_technique", "résumé non technique"]
    assert is_secondary("Beaumont_RNT_v2.pdf", patterns)
    assert is_secondary("Résumé non technique.pdf", patterns)
    assert is_secondary("04_resume_non_technique.pdf", patterns)
    assert not is_secondary("Tome_4_etude_impact.pdf", patterns)
    assert not is_secondary("x.pdf", [""])
    assert strip_accents("Résumé l’étude") == "Resume l'etude"


def test_lot_sans_sommaire_et_secondaires_plafonnes(env, monkeypatch):
    big = {"rerank_top_k": 50, "rerank_candidates": 50}
    free = retriever(env, RetrievalSpec(**big)).retrieve(QUERY)
    files = [p.unit.file_name for p in free.passages]
    assert files.count("reunion-3.transcript.md") == 2  # deux parents dans ce fichier

    spec = RetrievalSpec(**big, secondary_documents=["reunion-3"], secondary_cap=1)
    capped = retriever(env, spec).retrieve(QUERY)
    assert [p.unit.file_name for p in capped.passages].count("reunion-3.transcript.md") == 1
    assert capped.capped_secondary >= 1 and len(capped.passages) == len(free.passages) - 1
    # Plafond à 0 : pas de plafond.
    spec = RetrievalSpec(**big, secondary_documents=["reunion-3"], secondary_cap=0)
    assert retriever(env, spec).retrieve(QUERY).capped_secondary == 0

    monkeypatch.setattr(
        "luciole.retrieval.hybrid.is_table_of_contents", lambda text: "Inès" in text
    )
    result = retriever(env, RetrievalSpec(**big)).retrieve(QUERY)
    assert result.skipped_toc > 0 and all("Inès" not in p.hit.text for p in result.passages)
    off = retriever(env, RetrievalSpec(**big, exclude_toc=False)).retrieve(QUERY)
    assert off.skipped_toc == 0
