"""Banc d'essai : mesure la recherche et les réponses sur un jeu de référence.

    python -m luciole.evaluation.bench --profile mrae --reference reference.jsonl
    python -m luciole.evaluation.bench --profile mrae --reference reference.jsonl --answer

Pour chaque remarque, la recherche est limitée à son dossier. Mesures sur les
remarques qui ont des renvois attendus :

- **bon tome @k** : un des k premiers passages vient d'une pièce attendue ;
- **MRR tome** : moyenne de 1 / rang du premier passage de la bonne pièce ;
- **bonne page @k** : un des k premiers passages couvre une page attendue de sa
  pièce (à ``page_tolerance`` pages près).

Avec ``--answer`` (Ollama lancé), le LLM rédige chaque réponse : esquives et
verdicts sur toutes les remarques, **tome cité** sur celles qui ont des renvois,
et **tome fourni** : une pièce attendue figure parmi les passages donnés au LLM
(avec ``--agent``, ce sont ceux de la boucle, questions composées comprises ;
les mesures @k portent toujours sur la recherche de la seule remarque).
``--runs 2`` (défaut) refait tout deux fois : les remarques dont le résultat
change d'un passage à l'autre donnent le plancher de bruit.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from luciole.core.versions import Versions
from luciole.evaluation.reference import ReferenceItem, read_reference
from luciole.evaluation.spec import BenchSpec, MetricsSpec, ReferenceSpec, load_bench_spec
from luciole.generation.guards import SourcePages, lot_pages
from luciole.store.sqlite import Condition


@dataclass
class ItemScore:
    id: str
    has_expected: bool
    #: Pièce de chacun des passages rendus, dans l'ordre.
    parts: list[str | None] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    part_rank: int | None = None
    page_rank: int | None = None
    seconds: float = 0.0
    #: Réponse (avec ``--answer``).
    verdict: str | None = None
    evasive: bool | None = None
    cited_part: bool | None = None
    #: Une pièce attendue figure parmi les passages donnés au LLM (avec ``--answer``).
    given_part: bool | None = None
    #: Nombre de passages donnés au LLM.
    given: int | None = None
    #: Pièces nommées dans le texte de la réponse (« Tome 4 », « reunion-3 »).
    named_parts: list[str] = field(default_factory=list)
    #: Questions de recherche composées par la boucle (avec ``--agent``).
    questions: list[str] = field(default_factory=list)
    answer: str | None = None


def _part_of(store, document_id: str, part_field: str, cache: dict) -> str | None:
    if document_id not in cache:
        doc = store.get_document(document_id)
        value = None if doc is None else doc.metadata.get(part_field)
        cache[document_id] = None if value in (None, "") else str(int(str(value)))
    return cache[document_id]


def _covers(passage, pages: set[int], tolerance: int) -> bool:
    unit = passage.unit
    covered = lot_pages([SourcePages("x", unit.page_start, unit.page_end, unit.page_labels)]).get(
        "x", set()
    )
    return any(abs(a - b) <= tolerance for a in covered for b in pages)


def score_passages(
    item: ReferenceItem, passages, store, metrics: MetricsSpec, cache: dict
) -> ItemScore:
    expected = {e.part: set(e.pages) for e in item.expected}
    score = ItemScore(item.id, bool(expected))
    for rank, p in enumerate(passages, start=1):
        part = _part_of(store, p.unit.document_id, metrics.part_field, cache)
        score.parts.append(part)
        score.citations.append(p.citation)
        if part in expected:
            if score.part_rank is None:
                score.part_rank = rank
            if score.page_rank is None and _covers(p, expected[part], metrics.page_tolerance):
                score.page_rank = rank
    return score


def named_parts(text: str, reference: ReferenceSpec, dossier: str | None = None) -> list[str]:
    """Pièces nommées dans un texte (« Tome 4 », « reunion-3.transcript.md »).

    Avec ``dossier``, une pièce écrite dans le chemin d'un autre dossier
    (« geometres-blanc/reunion-2.transcript.md », un dossier comparé) ne
    compte pas."""
    found = set()
    for m in re.finditer(reference.part_pattern, text, re.I):
        token = re.split(r"[\s\[\](),;:]", text[: m.start()])[-1]
        if dossier is not None and "/" in token and token.rstrip("/").split("/")[-1] != dossier:
            continue
        found.add(str(int(m.group(1))))
    return sorted(found, key=int)


def cites_part(
    answer, item: ReferenceItem, reference: ReferenceSpec, parts: list, by_pages: bool = True
) -> bool:
    """La réponse renvoie à une pièce attendue : nommée dans le texte, ou (si
    ``by_pages``) une page citée d'un passage de cette pièce."""
    wanted = {e.part for e in item.expected}
    if set(named_parts(answer.answer, reference, item.dossier)) & wanted:
        return True
    return by_pages and any(
        c["cited"] and part in wanted for c, part in zip(answer.citations(), parts, strict=True)
    )


def composed_questions(trace: dict | None) -> list[str]:
    """Les questions de l'outil ``compose`` dans la trace d'une boucle."""
    for step in (trace or {}).get("steps", []):
        if step.get("type") == "compose":
            return list(step["detail"].get("questions", []))
    return []


def run_once(items, retriever, store, spec: BenchSpec, answerer=None, log=print) -> list[ItemScore]:
    top = max(spec.metrics.k)
    cache: dict = {}
    scores = []
    for n, item in enumerate(items, start=1):
        started = time.perf_counter()
        where = [Condition("dossier_id", "eq", item.dossier)]
        result = retriever.retrieve(item.question, conditions=where, top_k=top)
        if result.allowed_documents == 0:
            raise ValueError(f"{item.id} : aucun document pour le dossier « {item.dossier} »")
        score = score_passages(item, result.passages, store, spec.metrics, cache)
        if answerer is not None:
            answer = answerer.answer(item.question, conditions=where)
            # Les passages d'un autre dossier (comparables d'une boucle) ne
            # comptent ni comme pièce fournie ni comme pièce citée.
            parts = [
                _part_of(store, p.unit.document_id, spec.metrics.part_field, cache)
                if p.unit.dossier_id == item.dossier
                else None
                for p in answer.passages
            ]
            score.verdict = answer.verdict
            score.evasive = answer.guards.evasive
            score.answer = answer.answer
            score.given = len(parts)
            score.questions = composed_questions(answer.trace)
            score.named_parts = named_parts(answer.answer, spec.reference, item.dossier)
            if item.expected:
                score.cited_part = cites_part(
                    answer, item, spec.reference, parts, spec.metrics.cite_by_pages
                )
                score.given_part = bool({e.part for e in item.expected} & set(parts))
        score.seconds = round(time.perf_counter() - started, 2)
        scores.append(score)
        mark = "-" if not score.has_expected else ("ok" if score.part_rank == 1 else "··")
        log(f"  [{n}/{len(items)}] {item.id} {mark} ({score.seconds} s)")
    return scores


def summarize(scores: list[ItemScore], metrics: MetricsSpec) -> dict:
    graded = [s for s in scores if s.has_expected]
    n = len(graded)
    out: dict = {"remarques": len(scores), "avec_renvoi": n}
    for k in metrics.k:
        out[f"bon_tome@{k}"] = sum(s.part_rank is not None and s.part_rank <= k for s in graded)
        out[f"bonne_page@{k}"] = sum(s.page_rank is not None and s.page_rank <= k for s in graded)
    out["mrr_tome"] = round(sum(1 / s.part_rank for s in graded if s.part_rank) / n, 3) if n else 0
    answered = [s for s in scores if s.verdict is not None]
    if answered:
        out["esquives"] = sum(bool(s.evasive) for s in answered)
        out["verdicts"] = {
            v: sum(s.verdict == v for s in answered)
            for v in ("SUFFISANT", "PARTIEL", "INSUFFISANT")
        }
        out["tome_fourni"] = sum(bool(s.given_part) for s in answered if s.has_expected)
        out["tome_cite"] = sum(bool(s.cited_part) for s in answered if s.has_expected)
    out["secondes"] = round(sum(s.seconds for s in scores), 1)
    return out


def noise(runs: list[list[ItemScore]], metrics: MetricsSpec) -> dict:
    """Remarques dont le résultat change d'un passage à l'autre."""
    top = max(metrics.k)

    def outcome(s: ItemScore) -> tuple:
        hit = s.part_rank is not None and s.part_rank <= top
        page = s.page_rank is not None and s.page_rank <= top
        return (hit, page, s.evasive, s.verdict, s.cited_part)

    changed = sorted(
        {
            s.id
            for scores in runs[1:]
            for s, first in zip(scores, runs[0], strict=True)
            if outcome(s) != outcome(first)
        }
    )
    return {"passages": len(runs), "remarques_changeantes": changed}


def table(summary: dict, metrics: MetricsSpec, baseline: dict[str, str]) -> str:
    n = summary["avec_renvoi"]
    part, e = metrics.part_label, "e" if metrics.part_feminine else ""
    right = f"bon{'ne' if e else ''} {part}"
    lines = [f"Remarques : {summary['remarques']} (dont {n} avec renvoi aux pièces)"]
    for k in metrics.k:
        lines.append(
            f"  {right} @{k} : {summary[f'bon_tome@{k}']}/{n}"
            f"   bonne page @{k} : {summary[f'bonne_page@{k}']}/{n}"
        )
    lines.append(f"  MRR {part} : {summary['mrr_tome']}")
    if "esquives" in summary:
        v = summary["verdicts"]
        lines.append(f"  esquives : {summary['esquives']}/{summary['remarques']}")
        lines.append(f"  {part} fourni{e} au LLM : {summary['tome_fourni']}/{n}")
        lines.append(f"  {part} cité{e} dans la réponse : {summary['tome_cite']}/{n}")
        lines.append(
            f"  verdicts : {v['SUFFISANT']} suffisant, {v['PARTIEL']} partiel, "
            f"{v['INSUFFISANT']} insuffisant"
        )
    lines.append(f"  durée : {summary['secondes']} s")
    if baseline:
        lines.append("Référence : " + " ; ".join(f"{k} {v}" for k, v in baseline.items()))
    return "\n".join(lines)


def report(
    items, runs, spec: BenchSpec, *, profile: str, reference: str, models: dict, mode: str = "rag"
) -> dict:
    return {
        "profile": profile,
        "mode": mode,
        "reference": reference,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "versions": Versions().as_dict(),
        "models": models,
        "metrics": spec.metrics.model_dump(),
        "baseline": spec.baseline,
        "runs": [
            {"summary": summarize(scores, spec.metrics), "items": [asdict(s) for s in scores]}
            for scores in runs
        ],
        "noise": noise(runs, spec.metrics),
        "questions": {i.id: i.question for i in items},
    }


def parser():
    from luciole.retrieval import search

    cli = argparse.ArgumentParser(
        prog="python -m luciole.evaluation.bench",
        description="Banc d'essai : bon tome, bonne page, MRR ; esquives avec --answer.",
    )
    search.add_service_options(cli)
    cli.add_argument("--reference", required=True, help="jeu de référence JSONL (hors dépôt)")
    cli.add_argument("--answer", action="store_true", help="faire rédiger les réponses (Ollama)")
    cli.add_argument(
        "--agent", action="store_true", help="répondre par la boucle du profil (implique --answer)"
    )
    cli.add_argument("--runs", type=int, default=2, help="nombre de passages (défaut 2)")
    cli.add_argument("--only", action="append", default=[], help="identifiant de remarque")
    cli.add_argument("--report", help="rapport JSON (défaut : data/bench/<profil>-<date>.json)")
    return cli


def main(argv: list[str] | None = None, *, log=print, **services) -> int:
    import os

    from luciole.generation import ask
    from luciole.retrieval import search

    args = parser().parse_args(argv)
    if args.runs < 1:
        raise SystemExit("--runs doit être au moins 1")
    spec = load_bench_spec(Path(args.profiles_dir) / args.profile)
    items = read_reference(args.reference)
    if args.only:
        items = [i for i in items if i.id in set(args.only)]
        if not items:
            raise SystemExit("aucune remarque ne correspond à --only")
    args.question, args.where, args.top_k = "", [], None
    llm_factory = services.pop("llm_factory", None)
    mode = "boucle" if args.agent else "rag"
    if args.answer or args.agent:
        answerer, store = ask.build(args, llm_factory=llm_factory, **services)
        retriever = answerer.retriever
        llm = getattr(answerer.llm, "model", None)
        if args.agent:
            from luciole.agents.tools import build_agent
            from luciole.core.profile import load_profile

            loop = load_profile(Path(args.profiles_dir) / args.profile).loop
            answerer = build_agent(loop, answerer, tracer=answerer.tracer)
    else:
        answerer, llm = None, None
        retriever, store = search.build(args, **services)
    models = {
        "embedding": getattr(retriever.embedder, "name", None),
        "reranker": getattr(retriever.reranker, "name", None),
        "llm": llm,
    }
    runs = []
    try:
        for r in range(args.runs):
            log(f"Passage {r + 1}/{args.runs} ({mode}) : {len(items)} remarque(s)")
            runs.append(run_once(items, retriever, store, spec, answerer, log=log))
            log(table(summarize(runs[-1], spec.metrics), spec.metrics, {}))
    finally:
        store.close()
    out = report(
        items, runs, spec, profile=args.profile, reference=args.reference, models=models, mode=mode
    )
    data_dir = Path(os.environ.get("LUCIOLE_DATA_DIR") or "data")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(args.report or data_dir / "bench" / f"{args.profile}-{stamp}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    changed = out["noise"]["remarques_changeantes"]
    log("")
    log(table(out["runs"][-1]["summary"], spec.metrics, spec.baseline))
    log(
        f"Plancher de bruit : {len(changed)} remarque(s) changent d'un passage à l'autre"
        + (f" ({', '.join(changed)})" if changed else "")
    )
    log(f"Rapport : {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
