"""Ligne de commande : poser une question à la boucle d'un cas d'usage.

    python -m luciole.agents.run "question" --profile mrae

Affiche la réponse comme ``luciole.generation.ask``, puis le chemin suivi
dans la boucle, les règles qui ont matché et le nœud d'arrêt.
"""

from __future__ import annotations

import sys
from pathlib import Path

from luciole.agents.engine import AgentResult
from luciole.agents.tools import build_agent
from luciole.core.profile import load_profile
from luciole.generation import ask


def parser():
    cli = ask.parser()
    cli.prog = "python -m luciole.agents.run"
    cli.description = "Boucle du cas d'usage : recherche, rédaction, verdict, raffinement borné."
    return cli


def build(args, **services):
    """La boucle du profil et le magasin, à partir des options de ``ask``."""
    answerer, store = ask.build(args, **services)
    loop = load_profile(Path(args.profiles_dir) / args.profile).loop
    return build_agent(loop, answerer, tracer=answerer.tracer), store


def show(result: AgentResult) -> str:
    lines = [ask.show(result.answer), "", f"Chemin : {' > '.join(result.path)}"]
    for rule in result.matched_rules:
        lines.append(f"  règle de « {rule['node']} » : {rule['when']} -> {rule['then']}")
    lines.append(f"Arrêt : {result.stopped_reason} ({result.steps_used} étape(s))")
    return "\n".join(lines)


def main(argv: list[str] | None = None, **services) -> int:
    args = parser().parse_args(argv)
    agent, store = build(args, **services)
    try:
        result = agent.run(args.question, conditions=args.where)
    finally:
        store.close()
    print(show(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
