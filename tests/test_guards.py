"""Gardes déterministes reprises de ``query2`` (V5) : esquive, réponse qui
boucle, pages citées hors lot, nombres absents des extraits. Les quatre
premiers tests d'esquive sont ceux de V5 (``test_v4_corrections.py``)."""

from luciole.generation.guards import (
    GuardReport,
    SourcePages,
    cited_pages,
    is_degenerate,
    is_evasive,
    lot_pages,
    numbers_of,
    numbers_outside_extracts,
    pages_outside_lot,
)


# --- esquive (tests de V5) ------------------------------------------------
def test_esquive_forme_accentuee_detectee():
    text = "Les études ne précisent pas la distance aux habitations. " + "x " * 400
    assert is_evasive(text)


def test_esquive_aucun_element_accentue():
    assert is_evasive("Aucun élément du dossier n'aborde ce point. " + "y " * 400)


def test_reponse_sourcee_qui_concede_a_la_fin():
    body = "Pour rappel, le Tome 4 p. 12 indique la mesure ECO-E1. " * 20
    assert not is_evasive(body + " Toutefois le dossier ne précise pas la date.")


def test_apostrophe_typographique():
    assert is_evasive("Le dossier n’évoque aucun suivi. " + "z " * 400)


def test_esquive_courte_ou_sans_citation():
    assert is_evasive("")
    assert is_evasive("   ")
    # Motif tardif, mais réponse courte : esquive.
    assert is_evasive(
        "Le parc compte 8 éoliennes au sud. " * 3 + "Aucune information sur le bruit."
    )
    # Motif tardif, longue, mais aucune citation : esquive.
    long = "Le parc compte 8 éoliennes au sud du village. " * 30
    assert is_evasive(long + "Le dossier ne précise pas la date.")
    # Pas de motif : jamais une esquive.
    assert not is_evasive("Le Tome 4 p. 212 donne 650 m.")


def test_seuils_d_esquive_reglables():
    text = "Intro. " * 10 + "Le dossier ne précise pas la date. " + "Tome 4 p. 12. " * 60
    assert not is_evasive(text, max_position=0.05)
    assert is_evasive(text, max_position=0.5)


# --- réponse qui boucle ---------------------------------------------------
def test_reponse_degeneree():
    loop = "\n".join(["1. La mesure de bridage s'applique aux chiroptères."] * 9)
    assert is_degenerate(loop)
    varied = "\n".join(
        f"{i}. Ligne différente numéro {i} avec du contenu utile." for i in range(12)
    )
    assert not is_degenerate(varied)
    short = "\n".join(["- même ligne répétée plusieurs fois ici."] * 5)
    assert not is_degenerate(short)  # trop peu de lignes pour juger
    four = varied + "\n" + "\n".join(["Une ligne qui revient encore et encore ici."] * 4)
    assert is_degenerate(four)  # une ligne quatre fois


# --- pages ----------------------------------------------------------------
LOT = lot_pages(
    [
        SourcePages("Tome_4_etude_impact.pdf", 100, 101, ["212/400", "213/400"]),
        SourcePages("Tome_4_etude_impact.pdf", 150, 150),
        SourcePages("RNT.pdf", 12, 12, ["45"]),
        SourcePages("note.md"),  # sans page : ignoré
    ]
)


def test_pages_du_lot_avec_numeros_imprimes():
    assert LOT == {"tome_4_etude_impact.pdf": {100, 101, 150, 212, 213}, "rnt.pdf": {12, 45}}


def test_pages_hors_lot():
    ok = "Selon le Tome_4_etude_impact.pdf, p. 212, la distance est de 650 m."
    assert pages_outside_lot(ok, LOT) == []
    bad = "Selon le Tome_4_etude_impact.pdf, p. 300, la distance est de 650 m."
    assert pages_outside_lot(bad, LOT) == ["tome_4_etude_impact.pdf p. 300"]
    # « Tome 4 » désigne le fichier du tome 4.
    assert pages_outside_lot("Le Tome 4 (p. 45) indique…", LOT) == ["tome_4_etude_impact.pdf p. 45"]
    # Sans document reconnu : comparé à l'union du lot.
    assert pages_outside_lot("Voir page 45.", LOT) == []
    assert pages_outside_lot("Voir page 46.", LOT) == ["p. 46"]
    # Plage dont les deux bornes sont dans le lot : citations fusionnées.
    assert pages_outside_lot("RNT.pdf pages 12-45", LOT) == []
    assert pages_outside_lot("", LOT) == [] and pages_outside_lot("p. 3", {}) == []


def test_pages_citees():
    assert cited_pages("p. 12, pages 104-106 et page 7") == {7, 12, 104, 105, 106}
    assert cited_pages("pages 10-900") == {10}  # plage aberrante : la première page


# --- nombres --------------------------------------------------------------
def test_nombres_normalises():
    assert numbers_of("2 830 m, 0,80 et 1 200 €") == {"2830", "0.8", "1200"}


def test_nombres_hors_extraits():
    extracts = "La distance est de 2 830 m. Le niveau sonore atteint 35,0 dB(A)."
    answer = "1. La distance est de 2830 m (Tome 4 p. 212), et le bruit de 35 dB(A) ; 42 nuits."
    assert numbers_outside_extracts(answer, extracts) == ["42"]
    # Pages citées, numérotation et petits entiers ignorés.
    assert numbers_outside_extracts("2. Voir p. 999 : 3 éoliennes.", extracts) == []
    assert numbers_outside_extracts("Le seuil est 0,5.", extracts) == ["0,5"]
    assert numbers_outside_extracts("", extracts) == [] and numbers_outside_extracts("12", "") == []


def test_bilan_des_gardes():
    assert GuardReport().clean
    assert not GuardReport(numbers_outside=["42"]).clean
    assert GuardReport(evasive=True).to_dict()["evasive"] is True


def test_meme_nom_dans_deux_dossiers():
    lot = lot_pages(
        [
            SourcePages("atelier-dubois/reunion-2.transcript.md", 1, 1),
            SourcePages("cabinet-roche/reunion-2.transcript.md", 4, 4),
        ]
    )
    # Le chemin désigne un seul fichier : sa page 4 n'est pas dans le lot.
    assert pages_outside_lot("atelier-dubois/reunion-2.transcript.md, p. 4", lot) == [
        "atelier-dubois/reunion-2.transcript.md p. 4"
    ]
    assert pages_outside_lot("cabinet-roche/reunion-2.transcript.md, p. 4", lot) == []
    # Le nom seul est ambigu : comparé à tout le lot, sans choisir un dossier.
    assert pages_outside_lot("reunion-2.transcript.md, p. 4", lot) == []
    assert pages_outside_lot("reunion-2.transcript.md, p. 7", lot) == ["p. 7"]
