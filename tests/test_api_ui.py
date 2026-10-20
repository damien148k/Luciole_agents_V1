"""Chat servi par l'API (``luciole/ui/``), liste des cas d'usage, et boucle
relue à chaque question : une boucle modifiée sert à la question suivante sans
recharger les modèles."""

import re
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from luciole.api.app import UI_DIR, create_app
from luciole.core.profile import Loop
from luciole.generation.trace import JsonlTracer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_agent import RAG_LOOP  # noqa: E402
from test_answer import GOOD, QUESTION, ScriptedLLM, answerer  # noqa: E402
from test_index_pipeline import Setup  # noqa: E402

SHORT_LOOP = """
name: mrae
start: rechercher
nodes:
  rechercher: {type: retrieve, default: rediger}
  rediger: {type: synthesize, default: reponse}
  reponse: {type: terminal}
"""


@pytest.fixture()
def api(tmp_path):
    """Boucle « mrae » lue dans un fichier que le test réécrit."""
    loop_file = tmp_path / "loop.yaml"
    loop_file.write_text(yaml.safe_dump(RAG_LOOP.model_dump(exclude_defaults=True)), "utf-8")
    built = []

    def factory(use_case):
        built.append(use_case)
        s = Setup(tmp_path)
        s.pipeline.run(s.root)
        return answerer(s, ScriptedLLM(GOOD))

    def loops(use_case):
        if use_case == "ventes":
            raise ValueError("le nœud `a` renvoie vers `b`, qui n'existe pas")
        return Loop.model_validate(yaml.safe_load(loop_file.read_text("utf-8")))

    app = create_app(
        factory,
        tracer=JsonlTracer(tmp_path / "traces"),
        use_cases=lambda: ["mrae", "ventes"],
        loops=loops,
        describe=lambda name: {"title": name.upper(), "description": f"cas {name}"},
    )
    return TestClient(app), loop_file, built


def ask(client):
    return client.post("/api/agent/query", json={"question": QUESTION, "use_case": "mrae"})


def test_boucle_modifiee_servie_a_la_question_suivante(api):
    client, loop_file, built = api
    first = ask(client)
    assert first.status_code == 200, first.text
    assert first.json()["path"] == ["rechercher", "rediger", "verdict", "reponse"]
    loop_file.write_text(SHORT_LOOP, "utf-8")
    second = ask(client).json()
    assert second["path"] == ["rechercher", "rediger", "reponse"]
    assert built == ["mrae"]  # modèles et index chargés une seule fois


def test_boucle_invalide_refusee_puis_corrigee(api):
    client, loop_file, _ = api
    loop_file.write_text(SHORT_LOOP.replace("default: reponse", "default: nulle_part"), "utf-8")
    r = ask(client)
    assert r.status_code == 422 and "nulle_part" in r.json()["detail"]
    loop_file.write_text(SHORT_LOOP, "utf-8")
    assert ask(client).json()["stopped_reason"] == "reponse"


def test_liste_des_cas_d_usage(api):
    client, _, built = api
    r = client.get("/api/use-cases")
    assert r.status_code == 200
    mrae, ventes = r.json()
    assert mrae == {
        "name": "mrae",
        "title": "MRAE",
        "description": "cas mrae",
        "loop": "mrae",
        "loop_error": None,
    }
    assert ventes["loop"] is None and "n'existe pas" in ventes["loop_error"]
    assert built == []  # lister ne charge aucun modèle


def test_chat_servi_par_l_api(api):
    client, _, _ = api
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/ui/chat.html"
    page = client.get("/ui/chat.html")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    for asset in re.findall(r'(?:src|href)="([^"]+)"', page.text):
        assert client.get(f"/ui/{asset}").status_code == 200, asset
    assert "javascript" in client.get("/ui/chat.js").headers["content-type"]


def test_interface_hors_ligne():
    """Aucune ressource de l'interface n'est demandée à Internet."""
    files = sorted(UI_DIR.glob("*.*"))
    assert {f.suffix for f in files} >= {".html", ".js", ".css"}
    for f in files:
        assert not re.search(r"https?://", f.read_text("utf-8")), f.name


def test_profils_reels_decrits():
    """Sans doublure : titres et boucles lus dans ``configs/profiles``."""
    client = TestClient(create_app(lambda uc: None, use_cases=lambda: ["mrae", "ventes"]))
    names = {u["name"]: u for u in client.get("/api/use-cases").json()}
    assert names["mrae"]["loop"] == "mrae" and names["ventes"]["loop"] == "ventes"
    assert all(u["title"] and u["loop_error"] is None for u in names.values())
