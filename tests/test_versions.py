"""Les versions sont cohérentes entre le paquet et pyproject.toml."""

import tomllib
from pathlib import Path

import luciole
from luciole.core.versions import current


def test_version_du_paquet_suit_pyproject():
    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    assert luciole.__version__ == pyproject["project"]["version"]


def test_versions_attachees_aux_chunks():
    v = current().as_dict()
    assert set(v) == {"app", "index_schema", "parser", "embedding_profile"}
    assert all(
        isinstance(v[k], int) and v[k] >= 1 for k in ("index_schema", "parser", "embedding_profile")
    )
