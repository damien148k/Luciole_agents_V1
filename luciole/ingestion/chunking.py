"""Découpage d'un document en morceaux parents et enfants, mesurés en tokens.

Le découpeur travaille sur les nœuds du ``ParsedDocument`` (étape 1), pas sur
une chaîne de texte : il n'a rien à redeviner.

1. Les en-têtes, pieds et sauts de page sont écartés ; les titres ne sont
   pas des passages mais forment le **contexte** (« Titre > Section >
   Sous-section ») placé devant chaque morceau pour l'embedding.
2. Les nœuds consécutifs d'une même section forment des **parents** d'au
   plus ``parent_size_tokens`` ; un parent ne déborde jamais sur la section
   suivante.
3. Chaque parent est découpé en **enfants** d'au plus ``size_tokens``, avec
   un recouvrement d'environ ``overlap_pct`` % pris dans les derniers blocs
   de l'enfant précédent.
4. Un bloc trop long pour un enfant est coupé : un tableau par lignes (la
   ligne d'en-tête est répétée), un texte par phrases, puis par mots, puis
   par caractères en dernier recours.
5. **Garde** : aucun morceau, contexte compris, ne dépasse ``max_tokens``, ni
   la limite du tokenizer. Rien n'est tronqué en silence : un dépassement
   lève une erreur (porte de l'étape 2).

Toutes les longueurs sont comptées par le ``TokenCounter`` (le tokenizer du
modèle d'embedding), jamais en caractères.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from luciole.core.chunk import Chunk
from luciole.core.document import DocumentNode, PageLabel, ParsedDocument
from luciole.core.dossier import DocumentRecord
from luciole.core.profile import ChunkingSpec
from luciole.core.versions import current
from luciole.ingestion.tokenizer import TokenCounter

#: Nœuds qui ne sont jamais des passages.
_SKIPPED = {"header", "footer", "page_break", "title", "heading"}
_SEPARATOR = "\n\n"
_SENTENCE_RE = re.compile(r"(?<=[.!?…;:])\s+")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}")
_MARKUP_RE = re.compile(r"\*+|`+")


def _plain(heading: str) -> str:
    """Titre sans balises Markdown (« **Projet** » → « Projet »)."""
    return re.sub(r"\s+", " ", _MARKUP_RE.sub("", heading)).strip()


class ChunkLimitError(ValueError):
    """Un morceau dépasse la limite : erreur de programmation ou de réglage."""


@dataclass
class _Unit:
    """Un bloc de texte prêt à être assemblé (un nœud, ou un morceau de nœud)."""

    text: str
    tokens: int
    node: DocumentNode


@dataclass
class _Section:
    path: tuple[str, ...]
    units: list[_Unit] = field(default_factory=list)


def render_node(node: DocumentNode) -> str:
    """Texte d'un nœud tel qu'il entre dans un morceau : on garde qui parle
    dans une transcription et la puce d'une liste."""
    text = node.text.strip()
    if node.node_type == "speaker_turn":
        speaker = node.attributes.get("speaker")
        stamp = node.attributes.get("timestamp")
        prefix = f"[{stamp}] " if stamp else ""
        return f"{prefix}{speaker} : {text}" if speaker else text
    if node.node_type == "list_item":
        return f"- {text}"
    return text


class Chunker:
    def __init__(self, spec: ChunkingSpec, counter: TokenCounter) -> None:
        limit = counter.model_max_length
        if limit is not None and spec.max_tokens > limit:
            raise ValueError(
                f"chunking.max_tokens = {spec.max_tokens} dépasse la limite du tokenizer "
                f"{counter.name} ({limit})"
            )
        self.spec = spec
        self.counter = counter

    # ------------------------------------------------------------------
    def chunk(self, parsed: ParsedDocument, record: DocumentRecord) -> list[Chunk]:
        """Morceaux du document : parents puis enfants, dans l'ordre de lecture."""
        title = self._title(parsed, record)
        parents: list[Chunk] = []
        children: list[Chunk] = []
        seen_children: set[str] = set()
        for section in self._sections(parsed):
            context = self._context(title, section.path)
            context_tokens = self.counter.count(context) if context else 0
            # Le contexte compte dans la limite haute : il réduit la place du texte.
            room = self.spec.max_tokens - context_tokens - 2
            parent_budget = min(self.spec.parent_size_tokens, room)
            child_budget = min(self.spec.size_tokens, room)
            units = [
                piece
                for unit in section.units
                for piece in self._fit(unit, min(child_budget, parent_budget))
            ]
            for group in self._pack(units, parent_budget, overlap=0):
                parent = self._make(
                    parsed, record, "parent", len(parents), None, group, context, section.path
                )
                parents.append(parent)
                for sub in self._pack(group, child_budget, overlap=self.spec.overlap_tokens):
                    text = _SEPARATOR.join(u.text for u in sub)
                    if text in seen_children:  # passage répété mot pour mot : un seul
                        continue
                    seen_children.add(text)
                    children.append(
                        self._make(
                            parsed,
                            record,
                            "child",
                            len(children),
                            parent.chunk_id,
                            sub,
                            context,
                            section.path,
                        )
                    )
        chunks = parents + children
        self.check(chunks)
        return chunks

    def check(self, chunks: Iterable[Chunk]) -> None:
        """Garde de la porte de l'étape 2 : aucun dépassement, recompté."""
        for c in chunks:
            total = self.counter.count(c.embed_text)
            body = self.counter.count(c.text)
            limit = self.spec.size_tokens if c.kind == "child" else self.spec.parent_size_tokens
            if total > self.spec.max_tokens or body > limit or total != c.token_count:
                raise ChunkLimitError(
                    f"{c.chunk_id} : {total} tokens (passage {body}), limites "
                    f"{self.spec.max_tokens} et {limit}"
                )

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------
    @staticmethod
    def _title(parsed: ParsedDocument, record: DocumentRecord) -> str:
        for node in parsed.nodes:
            if node.node_type == "title" and node.text.strip():
                return node.text.strip()
        meta_title = parsed.metadata.get("title")
        if isinstance(meta_title, str) and meta_title.strip():
            return meta_title.strip()
        # « reunion-1.transcript.md » → « reunion-1 » ; « Tome 4 v1.2.pdf » → « Tome 4 v1.2 ».
        name = Path(record.file_name)
        while name.suffix[1:].isalpha() and len(name.suffix) <= 12:
            name = name.with_suffix("")
        return name.name

    def _sections(self, parsed: ParsedDocument) -> list[_Section]:
        sections: list[_Section] = []
        for node in parsed.nodes:
            if node.node_type in _SKIPPED:
                continue
            text = render_node(node)
            if not text:
                continue
            if not sections or sections[-1].path != node.section_path:
                sections.append(_Section(node.section_path))
            sections[-1].units.append(_Unit(text, self.counter.count(text), node))
        return sections

    def _context(self, title: str, path: tuple[str, ...]) -> str:
        """« Titre > Section > Sous-section », raccourci par le haut s'il prend
        plus du quart d'un enfant (la section la plus proche est gardée)."""
        budget = self.spec.size_tokens // 4
        title = _plain(title)
        parts = [_plain(p) for p in path]
        parts = [p for p in parts if p and p != title]
        while True:
            context = " > ".join([title, *parts])
            if self.counter.count(context) <= budget or not parts:
                break
            parts = parts[1:]
        if self.counter.count(context) > budget:
            context = self._split_text(context, budget)[0]
        return context

    # ------------------------------------------------------------------
    # Assemblage
    # ------------------------------------------------------------------
    def _fits(self, texts: list[str], budget: int) -> bool:
        return self.counter.count(_SEPARATOR.join(texts)) <= budget

    def _pack(self, units: list[_Unit], budget: int, *, overlap: int) -> list[list[_Unit]]:
        """Regroupe des blocs consécutifs sans dépasser ``budget`` ; chaque groupe
        reprend en tête les derniers blocs du précédent, dans ``overlap`` tokens."""
        groups: list[list[_Unit]] = []
        current: list[_Unit] = []
        fresh = 0  # blocs du groupe courant qui ne sont pas du recouvrement
        for unit in units:
            if self._fits([u.text for u in [*current, unit]], budget):
                current.append(unit)
                fresh += 1
                continue
            if fresh:
                groups.append(current)
            current = [*self._tail(current[-fresh:] if fresh else [], overlap), unit]
            while len(current) > 1 and not self._fits([u.text for u in current], budget):
                current.pop(0)
            fresh = 1
        if fresh:
            groups.append(current)
        return groups

    def _tail(self, units: list[_Unit], overlap: int) -> list[_Unit]:
        """Fin des ``units`` qui tient dans ``overlap`` tokens : des blocs entiers,
        ou les dernières phrases du dernier bloc s'il est plus long."""
        if overlap <= 0 or not units:
            return []
        tail: list[_Unit] = []
        used = 0
        for u in reversed(units):
            if used + u.tokens > overlap:
                break
            tail.insert(0, u)
            used += u.tokens
        if tail:
            return tail
        last = units[-1]
        kept = ""
        for sentence in reversed(_SENTENCE_RE.split(last.text)):
            candidate = f"{sentence} {kept}".strip()
            if self.counter.count(candidate) > overlap:
                break
            kept = candidate
        return [_Unit(kept, self.counter.count(kept), last.node)] if kept else []

    def _fit(self, unit: _Unit, budget: int) -> list[_Unit]:
        """Coupe un bloc trop long en morceaux qui tiennent dans ``budget``."""
        if unit.tokens <= budget:
            return [unit]
        if unit.node.node_type == "table":
            pieces = self._split_table(unit.text, budget)
        else:
            pieces = self._split_text(unit.text, budget)
        return [_Unit(p, self.counter.count(p), unit.node) for p in pieces]

    def _split_table(self, text: str, budget: int) -> list[str]:
        lines = [line for line in text.splitlines() if line.strip()]
        header: list[str] = []
        if len(lines) >= 2 and _TABLE_SEPARATOR_RE.match(lines[1]):
            header, lines = lines[:2], lines[2:]
        if header and not self._fits(["\n".join([*header, lines[0] if lines else ""])], budget):
            header = []  # en-tête trop large pour être répété : lignes seules
        pieces: list[str] = []
        rows: list[str] = []
        for line in lines:
            candidate = "\n".join([*header, *rows, line])
            if self.counter.count(candidate) <= budget:
                rows.append(line)
                continue
            if rows:
                pieces.append("\n".join([*header, *rows]))
            if self.counter.count("\n".join([*header, line])) <= budget:
                rows = [line]
            else:  # une seule ligne plus longue que le budget
                pieces.extend(self._split_text(line, budget))
                rows = []
        if rows:
            pieces.append("\n".join([*header, *rows]))
        return pieces

    def _split_text(self, text: str, budget: int) -> list[str]:
        """Phrases, puis mots, puis caractères : chaque morceau tient dans ``budget``."""
        text = text.strip()
        if not text:
            return []
        if self.counter.count(text) <= budget:
            return [text]
        for splitter, joiner in ((_SENTENCE_RE, " "), (re.compile(r"\s+"), " ")):
            parts = [p for p in splitter.split(text) if p]
            if len(parts) > 1:
                return self._join_parts(parts, joiner, budget)
        middle = len(text) // 2
        return self._split_text(text[:middle], budget) + self._split_text(text[middle:], budget)

    def _join_parts(self, parts: list[str], joiner: str, budget: int) -> list[str]:
        out: list[str] = []
        current = ""
        for part in parts:
            candidate = f"{current}{joiner}{part}" if current else part
            if self.counter.count(candidate) <= budget:
                current = candidate
                continue
            if current:
                out.append(current)
            if self.counter.count(part) <= budget:
                current = part
            else:
                out.extend(self._split_text(part, budget))
                current = ""
        if current:
            out.append(current)
        return out

    # ------------------------------------------------------------------
    def _make(
        self,
        parsed: ParsedDocument,
        record: DocumentRecord,
        kind: str,
        index: int,
        parent_id: str | None,
        units: list[_Unit],
        context: str,
        path: tuple[str, ...],
    ) -> Chunk:
        text = _SEPARATOR.join(u.text for u in units)
        nodes: list[DocumentNode] = []
        for u in units:
            if u.node not in nodes:
                nodes.append(u.node)
        pages = [n.page_start for n in nodes if n.page_start is not None]
        ends = [n.page_end for n in nodes if n.page_end is not None]
        labels: list[PageLabel] = []
        for n in nodes:
            labels.extend(lb for lb in parsed.labels_for(n) if lb not in labels)
        methods = sorted({n.method for n in nodes if n.method})
        prefix = "p" if kind == "parent" else "c"
        chunk = Chunk(
            chunk_id=f"{record.document_id}:{prefix}{index:05d}",
            kind=kind,
            parent_id=parent_id,
            index=index,
            document_id=record.document_id,
            dossier_id=record.dossier_id,
            relative_path=record.relative_path,
            file_name=record.file_name,
            doc_type=record.doc_type,
            content_hash=record.content_hash,
            text=text,
            context=context,
            token_count=1,
            page_start=min(pages) if pages else None,
            page_end=max(ends) if ends else None,
            page_labels=[lb.display() for lb in labels],
            section_path=path,
            node_ids=[n.node_id for n in nodes],
            methods=methods,
            metadata=dict(record.metadata),
            versions=current().as_dict(),
        )
        chunk.token_count = self.counter.count(chunk.embed_text)
        return chunk


# ---------------------------------------------------------------------------
# Ligne de commande : statistiques de découpage d'une racine de dossiers
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    from luciole.core.profile import load_profile
    from luciole.ingestion.dossiers import scan
    from luciole.ingestion.registry import default_registry
    from luciole.ingestion.tokenizer import load_tokenizer

    parser = argparse.ArgumentParser(
        prog="python -m luciole.ingestion.chunking",
        description="Découpe les documents d'une racine et vérifie les limites en tokens.",
    )
    parser.add_argument("root", help="dossier racine (un sous-répertoire par dossier)")
    parser.add_argument("--profile", required=True, help="nom du cas d'usage (mrae, ventes…)")
    parser.add_argument("--profiles-dir", default="configs/profiles")
    parser.add_argument("--tokenizer", help="nom ou chemin du tokenizer (défaut : le profil)")
    parser.add_argument("--show", type=int, default=0, help="afficher les N premiers enfants")
    args = parser.parse_args(argv)

    spec = load_profile(Path(args.profiles_dir) / args.profile).profile
    counter = load_tokenizer(args.tokenizer or spec.chunking.tokenizer)
    chunker = Chunker(spec.chunking, counter)
    print(
        f"Tokenizer : {counter.name} (limite du modèle : {counter.model_max_length or '?'}) ; "
        f"enfants ≤ {spec.chunking.size_tokens}, parents ≤ {spec.chunking.parent_size_tokens}, "
        f"limite haute {spec.chunking.max_tokens} tokens\n"
    )
    shown = 0
    worst = 0

    def on_document(record: DocumentRecord, parsed: ParsedDocument) -> None:
        nonlocal shown, worst
        chunks = chunker.chunk(parsed, record)
        kids = [c for c in chunks if c.kind == "child"]
        parents = len(chunks) - len(kids)
        top = max((c.token_count for c in chunks), default=0)
        worst = max(worst, top)
        labelled = sum(1 for c in kids if c.page_labels)
        print(
            f"{record.relative_path} : {parents} parent(s), {len(kids)} enfant(s), "
            f"max {top} tokens, {labelled}/{len(kids)} avec numéro de page imprimé"
        )
        for c in kids:
            if shown >= args.show:
                break
            shown += 1
            print(f"   --- {c.chunk_id} ({c.token_count} tokens) {c.citation()}")
            print(f"   {c.context}")
            print("   " + c.text[:300].replace("\n", "\n   "))

    scan(args.root, spec, default_registry(), on_document=on_document)
    print(f"\nPlus long morceau : {worst} tokens (limite {spec.chunking.max_tokens}) : OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
