"""Règles des boucles : expressions à liste blanche, évaluées en code.

Une règle ``when`` est une expression courte sur l'état de la boucle :

    verdict == 'INSUFFISANT' and refinements_left > 0
    count == 0
    verdict in ('PARTIEL', 'INSUFFISANT') or not evasive

Grammaire acceptée (rien d'autre) :

- champs : uniquement ceux de ``STATE_FIELDS`` ;
- valeurs : texte entre guillemets, nombres, ``true``/``false``/``True``/``False``,
  ``None``, listes ou n-uplets de valeurs ;
- comparaisons ``==  !=  <  <=  >  >=  in  not in`` (enchaînables) ;
- ``and``, ``or``, ``not`` et les parenthèses.

L'expression est lue par ``ast.parse`` puis **parcourue** par ce module : aucun
appel, attribut, indice, opération arithmétique ni ``eval`` (post-mortem
Belacom : des règles déclarées en YAML mais jamais exécutées ; ici chaque règle
est validée au chargement du profil et réellement évaluée). Une comparaison
d'ordre avec une valeur absente (``None``) est fausse, jamais une erreur.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass

# Champs de l'état qu'une règle peut lire (voir ``luciole.agents.engine.State``).
STATE_FIELDS: dict[str, str] = {
    "verdict": "SUFFISANT, PARTIEL ou INSUFFISANT (meilleure réponse jusqu'ici)",
    "confidence": "confiance de la meilleure réponse (score du reranker, 0 à 1)",
    "evasive": "la meilleure réponse esquive-t-elle la question ?",
    "count": "nombre de résultats du dernier outil (passages, documents…)",
    "refinements_left": "raffinements encore permis (décompté en code)",
    "steps_left": "nœuds encore permis avant le plafond `max_steps`",
    "step": "rang du nœud en cours (1 pour le premier)",
}

_CONSTANTS = {"true": True, "false": False}
_COMPARE = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}
_ORDER = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)


class RuleError(ValueError):
    """Expression de règle hors de la grammaire autorisée."""


@dataclass(frozen=True)
class Condition:
    """Une expression validée, prête à être évaluée sur un état."""

    source: str
    tree: ast.Expression
    fields: frozenset[str]

    def __call__(self, state: Mapping[str, object]) -> bool:
        return bool(_eval(self.tree.body, state))


def parse(source: str, fields: Mapping[str, object] = STATE_FIELDS) -> Condition:
    """Valide ``source`` et la rend évaluable ; lève ``RuleError`` sinon."""
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError as e:
        raise RuleError(f"règle illisible « {source} » : {e.msg}") from e
    used: set[str] = set()
    _check(tree.body, source, fields, used)
    return Condition(source, tree, frozenset(used))


def _check(node: ast.AST, source: str, fields: Mapping[str, object], used: set[str]) -> None:
    if isinstance(node, ast.BoolOp):
        for value in node.values:
            _check(value, source, fields, used)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        _check(node.operand, source, fields, used)
    elif isinstance(node, ast.Compare):
        for op in node.ops:
            if type(op) not in _COMPARE:
                raise RuleError(f"opérateur non permis dans « {source} »")
        for part in (node.left, *node.comparators):
            _check(part, source, fields, used)
    elif isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return
        if node.id not in fields:
            raise RuleError(
                f"champ inconnu « {node.id} » dans « {source} » "
                f"(champs permis : {', '.join(fields)})"
            )
        used.add(node.id)
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, str | int | float | bool | type(None)):
            raise RuleError(f"valeur non permise dans « {source} »")
    elif isinstance(node, ast.Tuple | ast.List):
        for elt in node.elts:
            if not isinstance(elt, ast.Constant):
                raise RuleError(f"une liste ne contient que des valeurs, dans « {source} »")
            _check(elt, source, fields, used)
    else:
        raise RuleError(
            f"« {source} » : seuls les champs, les valeurs, les comparaisons, "
            "and, or et not sont permis"
        )


def _eval(node: ast.AST, state: Mapping[str, object]) -> object:
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(_eval(v, state) for v in node.values)
        return any(_eval(v, state) for v in node.values)
    if isinstance(node, ast.UnaryOp):
        return not _eval(node.operand, state)
    if isinstance(node, ast.Compare):
        left = _eval(node.left, state)
        for op, right_node in zip(node.ops, node.comparators, strict=True):
            right = _eval(right_node, state)
            if isinstance(op, _ORDER) and (left is None or right is None):
                return False
            if not _COMPARE[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        return state.get(node.id)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple | ast.List):
        return tuple(_eval(e, state) for e in node.elts)
    raise RuleError("expression non validée")  # pragma: no cover - refusé par parse
