"""Jeu de référence : questions et renvois attendus, tirés de réponses réelles.

Une réponse du porteur de projet à un avis (DOCX) donne, pour chaque remarque,
la question posée et les pièces et pages de l'étude où se trouve la réponse.
Le jeu de référence en JSONL (une remarque par ligne) reste **hors du dépôt** :
il est tiré de documents réels.

    python -m luciole.evaluation.reference "Brissy_Réponse MRAe.docx" \\
        --profile mrae --dossier "Dossier Brissy" --prefix Brissy --out reference.jsonl

``--append`` ajoute les remarques d'une autre réponse au même fichier.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from luciole.evaluation.spec import ReferenceSpec, load_bench_spec


class Expected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    part: str = Field(description="Numéro de la pièce (tome)")
    pages: list[int] = Field(default_factory=list, description="Pages imprimées attendues")


class ReferenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    dossier: str = Field(description="dossier_id dans le magasin (sous-dossier de la racine)")
    question: str = Field(min_length=1)
    expected: list[Expected] = Field(default_factory=list)
    source: str = ""


def _blocks(path: Path) -> Iterator[tuple[str, list[str]]]:
    """Blocs du corps dans l'ordre : (« p », [texte]) ou (« table », [cellules])."""
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(str(path))
    for element in document.element.body.iterchildren():
        if element.tag.endswith("}p"):
            text = Paragraph(element, document).text.strip()
            if text:
                yield "p", [text]
        elif element.tag.endswith("}tbl"):
            cells: list[str] = []
            for row in Table(element, document).rows:
                for cell in row.cells:
                    text = cell.text.strip()
                    if text and (not cells or cells[-1] != text):  # cellules fusionnées
                        cells.append(text)
            if cells:
                yield "table", cells


def references(
    text: str, spec: ReferenceSpec, pending: str | None = None
) -> tuple[dict[str, set[int]], str | None]:
    """Pages attendues par pièce dans un bloc de texte, et la pièce laissée en
    attente pour le bloc suivant.

    Une page se rattache au renvoi de pièce le plus proche du même bloc
    (« Tome 4 … p. 12 » comme « page 59 du Tome 1 »). Un bloc qui nomme une
    pièce sans page (« Voir … Tome 4 – Milieu naturel : ») la passe au bloc
    suivant, qui donne les pages. Une page sans pièce est ignorée (annexes de la
    réponse, guides cités)."""
    parts = [(m.start(), m.end(), m.group(1)) for m in re.finditer(spec.part_pattern, text, re.I)]
    pages = list(re.finditer(spec.page_pattern, text, re.I))
    found: dict[str, set[int]] = {}
    if not parts and pending is None:
        return found, None
    for m in pages:
        start = int(m.group(1))
        end = int(m.group(2)) if m.lastindex and m.lastindex >= 2 and m.group(2) else start
        if end < start or end - start > spec.max_range:
            end = start
        numbers = set(range(start, end + 1))
        more = re.match(spec.page_more_pattern, text[m.end() :], re.I)
        if more:
            numbers.add(int(more.group(1)))

        def gap(part: tuple[int, int, str], m=m) -> int:
            return m.start() - part[1] if part[1] <= m.start() else part[0] - m.end()

        part = min(parts, key=lambda p: (abs(gap(p)), gap(p) < 0))[2] if parts else pending
        found.setdefault(str(int(part)), set()).update(numbers)
    waiting = parts[-1][2] if parts and not pages else None
    return found, waiting


def parse_response(
    path: str | Path, spec: ReferenceSpec, *, dossier: str, prefix: str
) -> list[ReferenceItem]:
    """Remarques d'une réponse DOCX, avec les renvois de la réponse à chacune."""
    path = Path(path)
    items: list[tuple[int, str, dict[str, set[int]]]] = []
    marker = re.compile(spec.remark_pattern, re.I)
    pending: str | None = None
    for kind, texts in _blocks(path):
        head = marker.match(texts[0]) if kind == "table" else None
        if head:
            question = "\n".join(texts[1:]).strip()
            items.append((int(head.group(1)), question, {}))
            pending = None
            continue
        if not items:
            continue  # préambule
        for text in texts:
            found, pending = references(text, spec, pending)
            for part, pages in found.items():
                items[-1][2].setdefault(part, set()).update(pages)
    out = []
    for number, question, found in items:
        if not question:
            raise ValueError(f"{path.name} : remarque {number} sans texte")
        out.append(
            ReferenceItem(
                id=f"{prefix}-R{number}",
                dossier=dossier,
                question=question,
                expected=[
                    Expected(part=part, pages=sorted(found[part]))
                    for part in sorted(found, key=int)
                ],
                source=path.name,
            )
        )
    ids = [i.id for i in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path.name} : numéros de remarque en double")
    return out


def read_reference(path: str | Path) -> list[ReferenceItem]:
    items = []
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if line.strip():
                try:
                    items.append(ReferenceItem.model_validate_json(line))
                except ValueError as e:
                    raise ValueError(f"{path}, ligne {n} : {e}") from e
    ids = [i.id for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} : identifiants en double")
    return items


def write_reference(items: list[ReferenceItem], path: str | Path, *, append: bool = False) -> None:
    path = Path(path)
    known = {i.id for i in read_reference(path)} if append and path.is_file() else set()
    clash = known & {i.id for i in items}
    if clash:
        raise ValueError(f"{path} contient déjà : {', '.join(sorted(clash))}")
    with path.open("a" if append else "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(
        prog="python -m luciole.evaluation.reference",
        description="Tire le jeu de référence d'une réponse à un avis (DOCX).",
    )
    cli.add_argument("response", help="réponse DOCX du porteur de projet")
    cli.add_argument("--profile", required=True)
    cli.add_argument("--profiles-dir", default="configs/profiles")
    cli.add_argument("--dossier", required=True, help="nom du dossier dans la racine indexée")
    cli.add_argument("--prefix", required=True, help="préfixe des identifiants (ex. Brissy)")
    cli.add_argument("--out", required=True, help="fichier JSONL (hors du dépôt)")
    cli.add_argument("--append", action="store_true")
    args = cli.parse_args(argv)
    spec = load_bench_spec(Path(args.profiles_dir) / args.profile).reference
    items = parse_response(args.response, spec, dossier=args.dossier, prefix=args.prefix)
    write_reference(items, args.out, append=args.append)
    with_refs = sum(bool(i.expected) for i in items)
    print(f"{len(items)} remarque(s), dont {with_refs} avec renvoi aux pièces → {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
