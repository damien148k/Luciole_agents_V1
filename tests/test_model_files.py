"""Modèles locaux à révision épinglée : récupération (faux téléchargeur),
vérification hors ligne, erreurs claires. Rien n'est téléchargé."""

import json
from pathlib import Path

import pytest

from luciole.core.model_files import (
    MARKER,
    ModelSpec,
    check,
    fetch,
    load_manifest,
    main,
    models_dir,
    require,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ModelSpec(
    repo="org/modele",
    revision="a" * 40,
    role="embedding",
    files=["config.json", "1_Pooling/config.json", "poids.bin"],
)


def fake_downloader(contents: dict[str, bytes] | None = None):
    calls = []

    def download(repo_id, revision, local_dir, allow_patterns):
        calls.append((repo_id, revision, list(allow_patterns)))
        for name in allow_patterns:
            data = (contents or {}).get(name, f"contenu de {name}".encode())
            if data is None:
                continue  # absent du dépôt à cette révision
            path = Path(local_dir) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    download.calls = calls
    return download


def test_liste_des_modeles_du_depot():
    manifest = load_manifest(ROOT / "configs" / "models.yaml")
    bge = manifest.get("bge-m3")
    assert bge.repo == "BAAI/bge-m3" and len(bge.revision) == 40
    assert bge.short_name == f"BAAI/bge-m3@{bge.revision[:7]}"
    # Leçon de V5 : pas de poids sparse ni ColBERT sans code qui les charge.
    assert not any("sparse" in f or "colbert" in f for f in bge.files)
    assert "tokenizer.json" in bge.files
    with pytest.raises(KeyError, match="connus : bge-m3"):
        manifest.get("inconnu")


def test_revision_doit_etre_un_commit_complet():
    with pytest.raises(ValueError):
        ModelSpec(repo="org/modele", revision="main", role="embedding", files=["a"])


def test_recuperation_puis_verification(tmp_path):
    download = fake_downloader()
    folder = fetch(SPEC, tmp_path, downloader=download)
    assert folder == tmp_path / "org" / "modele"
    assert download.calls == [("org/modele", "a" * 40, SPEC.files)]
    noted = json.loads((folder / MARKER).read_text(encoding="utf-8"))
    assert noted["revision"] == "a" * 40 and set(noted["files"]) == set(SPEC.files)
    assert check(SPEC, tmp_path) == []
    assert require(SPEC, tmp_path) == folder


def test_fichier_absent_du_depot_leve(tmp_path):
    with pytest.raises(FileNotFoundError, match="poids.bin"):
        fetch(SPEC, tmp_path, downloader=fake_downloader({"poids.bin": None}))
    assert not (tmp_path / "org" / "modele" / MARKER).exists()


def test_problemes_signales(tmp_path):
    assert check(SPEC, tmp_path)[0].startswith("dossier absent")
    folder = tmp_path / "org" / "modele"
    folder.mkdir(parents=True)
    assert MARKER in check(SPEC, tmp_path)[0], "copié à la main : révision inconnue"

    fetch(SPEC, tmp_path, downloader=fake_downloader())
    (folder / "poids.bin").write_bytes(b"autre chose, plus long")
    (folder / "config.json").unlink()
    other = SPEC.model_copy(update={"revision": "b" * 40})
    problems = check(other, tmp_path)
    assert any("révision" in p and "b" * 40 in p for p in problems)
    assert "fichier manquant : config.json" in problems
    assert "fichier modifié depuis la récupération : poids.bin" in problems


def test_modele_pas_pret_erreur_claire(tmp_path):
    with pytest.raises(FileNotFoundError, match="DEV.ps1 -Action models") as err:
        require(SPEC, tmp_path)
    assert "rien n'est téléchargé" in str(err.value)


def test_dossier_des_modeles(monkeypatch, tmp_path):
    monkeypatch.delenv("LUCIOLE_MODELS_DIR", raising=False)
    assert models_dir() == Path("models")
    monkeypatch.setenv("LUCIOLE_MODELS_DIR", str(tmp_path))
    assert models_dir() == tmp_path


def test_ligne_de_commande_check(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LUCIOLE_MODELS_DIR", str(tmp_path))
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        "models:\n  essai:\n    repo: org/modele\n    revision: " + "a" * 40 + "\n"
        "    role: embedding\n    files: [config.json, 1_Pooling/config.json, poids.bin]\n",
        encoding="utf-8",
    )
    assert main(["check", "--manifest", str(manifest)]) == 1
    assert "[ÉCHEC] essai" in capsys.readouterr().out
    fetch(SPEC, tmp_path, downloader=fake_downloader())
    assert main(["check", "essai", "--manifest", str(manifest)]) == 0
    assert "[OK] essai (org/modele@aaaaaaa" in capsys.readouterr().out


def test_ligne_de_commande_fetch_en_echec(tmp_path, monkeypatch, capsys):
    import luciole.core.model_files as mf

    def offline(spec, root=None, downloader=None):
        raise OSError("réseau injoignable")

    monkeypatch.setattr(mf, "fetch", offline)
    assert main(["fetch", "bge-m3", "--manifest", str(ROOT / "configs" / "models.yaml")]) == 1
    assert "[ÉCHEC] bge-m3 : réseau injoignable" in capsys.readouterr().out
