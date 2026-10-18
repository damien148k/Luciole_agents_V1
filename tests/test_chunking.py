"""Découpage en morceaux parents et enfants, mesurés en tokens.

Le tokenizer est injecté : ici un compteur factice (mots et ponctuation,
plus deux tokens spéciaux, comme BGE-M3), et un vrai tokenizer
``tokenizers`` fabriqué dans le test pour le chargement depuis le disque.
Rien n'est téléchargé. Le vrai tokenizer BGE-M3 est vérifié dans
``test_chunking_bge_m3.py`` (marqueur ``integration``).
"""

import json
import re
import sys
from pathlib import Path

import pytest

from luciole.core.chunk import Chunk
from luciole.core.document import NodeBuilder, PageInfo, PageLabel, ParsedDocument
from luciole.core.dossier import DocumentRecord
from luciole.core.profile import ChunkingSpec, load_profile
from luciole.ingestion.chunking import Chunker, ChunkLimitError, main
from luciole.ingestion.dossiers import scan
from luciole.ingestion.pdf import PdfParser
from luciole.ingestion.registry import default_registry
from luciole.ingestion.tokenizer import candidate_paths, load_tokenizer

ROOT = Path(__file__).resolve().parents[1]
VENTES_DIR = ROOT / "tests" / "corpus" / "ventes"
VENTES = load_profile(ROOT / "configs" / "profiles" / "ventes").profile

sys.path.insert(0, str(Path(__file__).resolve().parent / "corpus" / "pdf"))
from build_pdf_corpus import TWO_UP_LEFT, TWO_UP_RIGHT, build_all  # noqa: E402

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


class FakeCounter:
    """Un token par mot ou signe de ponctuation, plus [CLS] et [SEP]."""

    name = "faux"

    def __init__(self, model_max_length: int | None = 8192) -> None:
        self.model_max_length = model_max_length

    def count(self, text: str) -> int:
        return len(_TOKEN_RE.findall(text)) + 2


SMALL = ChunkingSpec(size_tokens=64, overlap_pct=20, parent_size_tokens=160, max_tokens=200)


def _record(name: str = "doc.md", dossier: str = "d", **metadata) -> DocumentRecord:
    return DocumentRecord(
        document_id="0" * 32,
        dossier_id=dossier,
        relative_path=f"{dossier}/{name}",
        file_name=name,
        doc_type="md",
        content_hash="h",
        size_bytes=1,
        page_count=1,
        metadata=metadata,
    )


def _doc(builder: NodeBuilder, pages: list[PageInfo] | None = None) -> ParsedDocument:
    return ParsedDocument(
        file_name="doc.md",
        source_path="doc.md",
        doc_type="md",
        pages=pages or [PageInfo(number=1)],
        nodes=builder.nodes(),
    )


def _sentences(n: int, word: str = "mesure") -> str:
    return " ".join(f"Phrase {i} sur la {word} du bruit au point fixe." for i in range(n))


@pytest.fixture(scope="module")
def pdfs(tmp_path_factory) -> dict[str, Path]:
    return build_all(tmp_path_factory.mktemp("pdf"))


def _assert_gate(chunks: list[Chunk], spec: ChunkingSpec, counter=None) -> None:
    """Porte de l'étape 2 : aucun morceau ne dépasse la limite, recompté ici."""
    counter = counter or FakeCounter()
    assert chunks
    for c in chunks:
        assert counter.count(c.embed_text) == c.token_count <= spec.max_tokens
        limit = spec.size_tokens if c.kind == "child" else spec.parent_size_tokens
        assert counter.count(c.text) <= limit, c.chunk_id


# ---------------------------------------------------------------------------
# Réglages
# ---------------------------------------------------------------------------
def test_reglages_par_defaut_et_recouvrement():
    spec = ChunkingSpec()
    assert (spec.size_tokens, spec.parent_size_tokens, spec.max_tokens) == (512, 1536, 2500)
    assert spec.tokenizer == "BAAI/bge-m3"
    assert ChunkingSpec(size_tokens=400, overlap_pct=10).overlap_tokens == 40


@pytest.mark.parametrize(
    "values",
    [
        {"size_tokens": 600, "parent_size_tokens": 500},
        {"parent_size_tokens": 3000, "max_tokens": 2500},
    ],
)
def test_tailles_incoherentes_refusees(values):
    with pytest.raises(ValueError, match="size_tokens <= parent_size_tokens <= max_tokens"):
        ChunkingSpec(**values)


def test_limite_au_dela_du_modele_refusee():
    with pytest.raises(ValueError, match="dépasse la limite du tokenizer"):
        Chunker(ChunkingSpec(max_tokens=8192), FakeCounter(model_max_length=512))


# ---------------------------------------------------------------------------
# Corpus de réunions fictives
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ventes_chunks() -> dict[str, list[Chunk]]:
    chunker = Chunker(SMALL, FakeCounter())
    out: dict[str, list[Chunk]] = {}
    scan(
        VENTES_DIR,
        VENTES,
        default_registry(),
        on_document=lambda rec, parsed: out.setdefault(
            rec.relative_path, chunker.chunk(parsed, rec)
        ),
    )
    return out


def test_reunions_decoupees_sous_la_limite(ventes_chunks):
    assert len(ventes_chunks) == 5
    for chunks in ventes_chunks.values():
        _assert_gate(chunks, SMALL)


def test_parents_et_enfants_lies(ventes_chunks):
    chunks = ventes_chunks["atelier-dubois/reunion-3.transcript.md"]
    parents = {c.chunk_id: c for c in chunks if c.kind == "parent"}
    children = [c for c in chunks if c.kind == "child"]
    assert parents and children
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for child in children:
        parent = parents[child.parent_id]
        # Chaque bloc de l'enfant est dans son parent : le parent l'entoure.
        assert set(child.node_ids) <= set(parent.node_ids)
        for block in child.text.split("\n\n"):
            assert block in parent.text
    assert [c.index for c in children] == list(range(len(children)))


def test_tours_de_parole_gardent_qui_parle(ventes_chunks):
    text = "\n".join(c.text for c in ventes_chunks["atelier-dubois/reunion-3.transcript.md"])
    assert "Camille Martin : " in text
    assert re.search(r"\[\d{2}:\d{2}(:\d{2})?\] ", text)


def test_tracabilite_dans_chaque_morceau(ventes_chunks):
    c = next(c for c in ventes_chunks["cabinet-roche/reunion-1.transcript.md"] if c.kind == "child")
    assert c.dossier_id == "cabinet-roche" and c.file_name == "reunion-1.transcript.md"
    assert len(c.content_hash) == 64 and c.document_id in c.chunk_id
    assert c.metadata["numero_reunion"] == 1 and c.metadata["date"] == "2026-02-12"
    assert c.methods == ["transcript"] and c.node_ids
    assert set(c.versions) == {"app", "index_schema", "parser", "embedding_profile"}
    # Pas de titre de niveau 1 : le nom du fichier, puis le titre de la réunion.
    assert c.context.startswith("reunion-1 > Réunion 1 — découverte")


# ---------------------------------------------------------------------------
# Règles de découpage
# ---------------------------------------------------------------------------
def test_un_enfant_ne_deborde_pas_sur_la_section_suivante():
    b = NodeBuilder()
    b.add("title", "Étude acoustique")
    b.add("heading", "Méthode", level=2)
    b.add("paragraph", "Six points fixes instrumentés.")
    b.add("heading", "Résultats", level=2)
    b.add("paragraph", "Émergences conformes de jour.")
    chunks = Chunker(SMALL, FakeCounter()).chunk(_doc(b), _record())
    children = [c for c in chunks if c.kind == "child"]
    assert [(c.context, c.text) for c in children] == [
        ("Étude acoustique > Méthode", "Six points fixes instrumentés."),
        ("Étude acoustique > Résultats", "Émergences conformes de jour."),
    ]
    assert [c.section_path for c in children] == [
        ("Étude acoustique", "Méthode"),
        ("Étude acoustique", "Résultats"),
    ]


def test_recouvrement_entre_enfants_voisins():
    b = NodeBuilder()
    for i in range(12):
        b.add("paragraph", f"Bloc {i} : le vent souffle du sud-ouest à six mètres par seconde.")
    doc = _doc(b)
    wide = ChunkingSpec(size_tokens=64, overlap_pct=40, parent_size_tokens=160, max_tokens=200)
    with_overlap = [
        c for c in Chunker(wide, FakeCounter()).chunk(doc, _record()) if c.kind == "child"
    ]
    pairs = [
        (p, n)
        for p, n in zip(with_overlap, with_overlap[1:], strict=False)
        if p.parent_id == n.parent_id
    ]
    assert pairs
    for prev, nxt in pairs:
        assert prev.text.split("\n\n")[-1] == nxt.text.split("\n\n")[0]
    # Un dernier bloc plus long que le recouvrement : on reprend ses dernières phrases.
    b = NodeBuilder()
    for i in range(4):
        b.add("paragraph", _sentences(4, word=f"série {i}"))
    children = [
        c for c in Chunker(wide, FakeCounter()).chunk(_doc(b), _record()) if c.kind == "child"
    ]
    pairs = [
        (p, n) for p, n in zip(children, children[1:], strict=False) if p.parent_id == n.parent_id
    ]
    assert pairs
    for prev, nxt in pairs:
        head = nxt.text.split("\n\n")[0]
        assert prev.text.endswith(head) and head != prev.text
    no_overlap = ChunkingSpec(size_tokens=64, overlap_pct=0, parent_size_tokens=160, max_tokens=200)
    children = [
        c for c in Chunker(no_overlap, FakeCounter()).chunk(doc, _record()) if c.kind == "child"
    ]
    blocks = [blk for c in children for blk in c.text.split("\n\n")]
    assert len(blocks) == len(set(blocks)) == 12, "sans recouvrement, chaque bloc une fois"


def test_long_paragraphe_coupe_aux_phrases():
    b = NodeBuilder()
    b.add("paragraph", _sentences(30))
    chunks = Chunker(SMALL, FakeCounter()).chunk(_doc(b), _record())
    _assert_gate(chunks, SMALL)
    children = [c for c in chunks if c.kind == "child"]
    assert len(children) > 3
    assert all(c.text.endswith(".") for c in children), "coupé entre deux phrases"


def test_mot_demesure_coupe_en_dernier_recours():
    b = NodeBuilder()
    blob = "".join(f"{i:04d}" for i in range(1250))  # 5000 caractères sans espace
    b.add("paragraph", blob)
    spec = ChunkingSpec(size_tokens=64, overlap_pct=0, parent_size_tokens=64, max_tokens=80)

    class CharCounter(FakeCounter):  # un token par tranche de 10 caractères
        def count(self, text: str) -> int:
            return len(text) // 10 + 2

    chunks = Chunker(spec, CharCounter()).chunk(_doc(b), _record())
    _assert_gate(chunks, spec, CharCounter())
    assert "".join(c.text for c in chunks if c.kind == "child") == blob


def test_tableau_coupe_par_lignes_avec_en_tete_repete():
    rows = "\n".join(f"| {i} m/s | {30 + i},5 | {31 + i},0 |" for i in range(40))
    b = NodeBuilder()
    b.add("table", f"| Vent | PF1 | PF2 |\n|---|---|---|\n{rows}")
    chunks = Chunker(SMALL, FakeCounter()).chunk(_doc(b), _record())
    _assert_gate(chunks, SMALL)
    children = [c for c in chunks if c.kind == "child"]
    assert len(children) > 2
    for c in children:
        assert c.text.startswith("| Vent | PF1 | PF2 |\n|---|---|---|\n")
    body = [line for c in children for line in c.text.splitlines()[2:]]
    assert set(body) == set(rows.splitlines()), "aucune ligne perdue"


def test_contexte_raccourci_par_le_haut():
    b = NodeBuilder()
    b.add("title", "Étude")
    for level in range(2, 7):
        b.add("heading", f"Section de niveau {level} " + "très longue " * 3, level=level)
    b.add("paragraph", "Le texte.")
    [child] = [
        c for c in Chunker(SMALL, FakeCounter()).chunk(_doc(b), _record()) if c.kind == "child"
    ]
    assert FakeCounter().count(child.context) <= SMALL.size_tokens // 4
    assert child.context.startswith("Étude > ") and "niveau 6" in child.context
    assert "niveau 2" not in child.context


def test_passages_vides_ou_repetes_ecartes():
    b = NodeBuilder()
    b.add("header", "Projet Vent du Nord", page=1)
    b.add("paragraph", "Même avertissement.", page=1)
    b.add("paragraph", "   ", page=1)
    b.add("page_break", "", page=1)
    b.add("paragraph", "Même avertissement.", page=2)
    b.add("footer", "Page 2 / 2", page=2)
    doc = _doc(b, [PageInfo(number=1), PageInfo(number=2)])
    spec = ChunkingSpec(size_tokens=64, overlap_pct=0, parent_size_tokens=64, max_tokens=80)
    chunks = Chunker(spec, FakeCounter()).chunk(doc, _record())
    children = [c for c in chunks if c.kind == "child"]
    assert [c.text for c in children] == ["Même avertissement.\n\nMême avertissement."]
    assert all("Projet Vent" not in c.text and "Page 2" not in c.text for c in chunks)


def test_garde_detecte_un_morceau_trop_long():
    b = NodeBuilder()
    b.add("paragraph", "Court.")
    chunker = Chunker(SMALL, FakeCounter())
    [parent, child] = chunker.chunk(_doc(b), _record())
    tampered = child.model_copy(update={"text": _sentences(20)})
    with pytest.raises(ChunkLimitError, match=child.chunk_id):
        chunker.check([parent, tampered])


# ---------------------------------------------------------------------------
# PDF : porte de l'étape 2 et numéros de page imprimés
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name", ["simple.pdf", "a3_dense.pdf", "dense_table.pdf", "tables.pdf", "two_up_rotated.pdf"]
)
def test_pdf_synthetiques_sous_la_limite(pdfs, name):
    parsed = PdfParser().parse(pdfs[name])
    chunks = Chunker(SMALL, FakeCounter()).chunk(parsed, _record(name))
    _assert_gate(chunks, SMALL)


def test_entetes_et_pieds_hors_des_morceaux(pdfs):
    parsed = PdfParser().parse(pdfs["simple.pdf"])
    chunks = Chunker(SMALL, FakeCounter()).chunk(parsed, _record("simple.pdf"))
    assert all("Tome 4" not in c.text and "/ 3" not in c.text for c in chunks)
    second = next(c for c in chunks if c.page_start == 2)
    assert second.page_labels == ["2/3"]
    assert second.citation() == "d/simple.pdf, page 2/3 (PDF p. 2)"
    assert second.context == "Résumé non technique > 2. Chapitre 2"


def test_numero_imprime_de_la_bonne_moitie(pdfs):
    parsed = PdfParser().parse(pdfs["two_up_rotated.pdf"])
    by_text = {n.text: n for n in parsed.nodes}
    left = by_text[f"{TWO_UP_LEFT[1]} (page 3)"]
    right = by_text[f"{TWO_UP_RIGHT[1]} (page 3)"]
    assert (left.attributes["side"], right.attributes["side"]) == ("left", "right")
    assert [lb.display() for lb in parsed.labels_for(left)] == ["12/47"]
    assert [lb.display() for lb in parsed.labels_for(right)] == ["13/47"]
    chunks = Chunker(SMALL, FakeCounter()).chunk(parsed, _record("two_up_rotated.pdf"))
    for c in chunks:  # un morceau ne cite que des numéros de ses propres pages
        own = {
            lb.display()
            for p in range(c.page_start, c.page_end + 1)
            for lb in parsed.pages[p - 1].labels
        }
        assert c.page_labels and set(c.page_labels) <= own


def test_labels_par_cote_dans_le_modele():
    b = NodeBuilder()
    b.add("paragraph", "À gauche.", page=1, attributes={"side": "left"})
    b.add("paragraph", "Sans côté connu.", page=1)
    labels = [
        PageLabel(label="7", total=47, side="left"),
        PageLabel(label="8", total=47, side="right"),
    ]
    doc = _doc(b, [PageInfo(number=1, width=1191, labels=labels)])
    left, unknown = doc.nodes
    assert [lb.display() for lb in doc.labels_for(left)] == ["7/47"]
    assert [lb.display() for lb in doc.labels_for(unknown)] == ["7/47", "8/47"]


# ---------------------------------------------------------------------------
# Tokenizer lu sur le disque, jamais téléchargé
# ---------------------------------------------------------------------------
def _save_tokenizer(folder: Path, max_length: int | None = 512) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers

    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(
        ["le vent souffle du sud", "réunion de suivi"],
        trainers.WordLevelTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]"]),
    )
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    folder.mkdir(parents=True, exist_ok=True)
    tok.save(str(folder / "tokenizer.json"))
    if max_length is not None:
        (folder / "tokenizer_config.json").write_text(json.dumps({"model_max_length": max_length}))
    return folder / "tokenizer.json"


def test_tokenizer_trouve_dans_le_cache_hugging_face(tmp_path, monkeypatch):
    snapshot = tmp_path / "hub" / "models--org--modele" / "snapshots" / "abc123"
    _save_tokenizer(snapshot)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.delenv("LUCIOLE_MODELS_DIR", raising=False)
    counter = load_tokenizer("org/modele")
    assert counter.name == "org/modele" and counter.model_max_length == 512
    assert counter.count("le vent souffle") == 5, "3 mots + [CLS] + [SEP]"
    assert counter.count("") == 2


def test_tokenizer_par_chemin_ou_dossier_des_modeles(tmp_path, monkeypatch):
    path = _save_tokenizer(tmp_path / "BAAI" / "bge-m3", max_length=None)
    assert load_tokenizer(str(path)).model_max_length is None
    monkeypatch.setenv("LUCIOLE_MODELS_DIR", str(tmp_path))
    assert load_tokenizer("BAAI/bge-m3").count("réunion de suivi") == 5


def test_tokenizer_absent_erreur_claire(tmp_path, monkeypatch):
    # Isoler tous les emplacements, y compris le cache Hugging Face du dossier
    # personnel : sur une machine où BGE-M3 est déjà téléchargé, il serait trouvé.
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "vide"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HOME", str(tmp_path / "maison"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "maison"))
    monkeypatch.setenv("LUCIOLE_MODELS_DIR", str(tmp_path / "modeles"))
    with pytest.raises(FileNotFoundError, match="rien n'est téléchargé") as err:
        load_tokenizer("BAAI/bge-m3")
    assert str(tmp_path / "modeles") in str(err.value)
    assert len(candidate_paths("BAAI/bge-m3")) == 2


def test_ligne_de_commande(tmp_path, capsys):
    path = _save_tokenizer(tmp_path / "tok", max_length=8192)
    code = main(
        [
            str(VENTES_DIR),
            "--profile",
            "ventes",
            "--profiles-dir",
            str(ROOT / "configs" / "profiles"),
            "--tokenizer",
            str(path),
            "--show",
            "1",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "atelier-dubois/reunion-1.transcript.md : 1 parent(s)" in out
    assert "Plus long morceau" in out and ": OK" in out
