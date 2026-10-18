"""Ligne de commande : poser une question et obtenir une réponse sourcée.

    python -m luciole.generation.ask "question" --profile mrae

Affiche la réponse, le verdict, la confiance, les sources et l'identifiant de
trace. Ollama doit tourner avec le modèle du profil matériel
(``LUCIOLE_OLLAMA_URL``, défaut ``http://127.0.0.1:11434``).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from luciole.generation.answer import Answer, Answerer
from luciole.generation.llm import OllamaLLM
from luciole.generation.prompts import load_prompts
from luciole.generation.trace import JsonlTracer
from luciole.retrieval import search


def parser():
    cli = search.parser()
    cli.prog = "python -m luciole.generation.ask"
    cli.description = "Réponse sourcée : recherche, rédaction par le LLM local, gardes, verdict."
    return cli


def build(args, *, llm_factory: Callable | None = None, tracer=None, **services):
    """Answerer et magasin à partir des options ; ``llm_factory(hardware,
    generation)`` et les services remplacent Ollama et les modèles en test."""
    from luciole.core.profile import load_profile
    from luciole.index.pipeline import _hardware

    retriever, store = search.build(args, **services)
    bundle = load_profile(Path(args.profiles_dir) / args.profile)
    hardware = _hardware(args.hardware, Path(args.hardware_dir))
    gen = bundle.profile.generation
    if llm_factory is None:

        def llm_factory(hw, generation):
            return OllamaLLM(hw.llm_model, num_ctx=hw.llm_num_ctx)

    answerer = Answerer(
        bundle.profile,
        load_prompts(bundle.path),
        retriever,
        llm_factory(hardware, gen),
        hardware,
        tracer=tracer if tracer is not None else JsonlTracer(),
    )
    return answerer, store


def show(result: Answer) -> str:
    lines = [
        result.answer,
        "",
        f"Verdict : {result.verdict} (confiance {result.confidence:.2f})",
    ]
    flags = [
        name
        for name, on in (
            ("esquive", result.guards.evasive),
            ("tronquée", result.guards.truncated),
            ("dégénérée", result.guards.degenerate),
        )
        if on
    ]
    if flags:
        lines.append("Gardes : " + ", ".join(flags))
    for c in result.citations():
        mark = "citée" if c["cited"] else "fournie"
        lines.append(f"  - {c['citation']} ({mark})")
    lines.append(f"Trace : {result.trace_id}")
    return "\n".join(lines)


def main(argv: list[str] | None = None, **services) -> int:
    args = parser().parse_args(argv)
    answerer, store = build(args, **services)
    try:
        result = answerer.answer(args.question, conditions=args.where)
    finally:
        store.close()
    print(show(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
