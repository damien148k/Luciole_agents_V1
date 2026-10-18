"""HTML et courriels (EML) sur le modèle documentaire.

- HTML : titres h1–h6, paragraphes, listes, tableaux ; navigation, scripts,
  styles et pieds de page sont écartés.
- EML : les en-têtes utiles (de, à, objet, date) vont dans les métadonnées
  et dans un premier bloc, le corps texte (ou HTML converti) suit. Les
  courriels sont un type de document du cas « commerciaux ».
"""

from __future__ import annotations

import re
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from pathlib import Path

from bs4 import BeautifulSoup

from luciole.core.document import NodeBuilder, PageInfo, ParsedDocument
from luciole.ingestion.base import Parser
from luciole.ingestion.office import _md_table

_DROP_TAGS = ("script", "style", "nav", "footer", "header", "noscript", "svg", "form")
_BLOCK_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table", "pre", "blockquote")


def _clean(text: str) -> str:
    return " ".join(text.split())


def html_to_nodes(b: NodeBuilder, html: str, *, page: int = 1, method: str = "plain") -> int:
    """Ajoute les blocs d'un document HTML ; renvoie le nombre de caractères utiles."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    root = soup.body or soup
    chars = 0
    title_seen = False
    for el in root.find_all(_BLOCK_TAGS):
        # Un bloc imbriqué dans un bloc déjà traité (li dans table…) est ignoré ici.
        if el.find_parent(["table", "li", "blockquote"]) is not None and el.name != "table":
            continue
        if el.name == "table":
            rows = [
                [_clean(c.get_text(" ")) for c in tr.find_all(["th", "td"])]
                for tr in el.find_all("tr")
            ]
            rows = [r for r in rows if any(r)]
            md = _md_table(rows)
            if md:
                chars += len(md)
                b.add("table", md, page=page, method=method, attributes={"rows": len(rows)})
            continue
        text = _clean(el.get_text(" "))
        if not text:
            continue
        chars += len(text)
        if el.name[0] == "h" and el.name[1:].isdigit():
            level = int(el.name[1])
            if level == 1 and not title_seen:
                b.add("title", text, page=page, level=1, method=method)
                title_seen = True
            else:
                b.add("heading", text, page=page, level=max(2, level), method=method)
        elif el.name == "li":
            b.add("list_item", text, page=page, method=method)
        elif el.name == "pre":
            b.add("code", el.get_text(), page=page, method=method)
        else:
            b.add("paragraph", text, page=page, method=method)
    return chars


class HtmlParser(Parser):
    extensions = (".html", ".htm")

    def parse(self, path: str | Path) -> ParsedDocument:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = path.read_bytes()
        html = raw.decode("utf-8", errors="replace")
        soup_title = BeautifulSoup(html, "html.parser").title
        b = NodeBuilder()
        chars = html_to_nodes(b, html)
        return ParsedDocument(
            file_name=path.name,
            source_path=str(path),
            doc_type="html",
            metadata={"title": _clean(soup_title.get_text()) if soup_title else None},
            pages=[
                PageInfo(
                    number=1,
                    method="plain" if chars else None,
                    quality=1.0 if chars else 0.0,
                    char_count=chars,
                )
            ],
            nodes=b.nodes(),
        )


class EmlParser(Parser):
    extensions = (".eml",)

    def parse(self, path: str | Path) -> ParsedDocument:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        msg = BytesParser(policy=policy.default).parsebytes(path.read_bytes())

        def header(name: str) -> str | None:
            v = msg.get(name)
            return _clean(str(v)) if v else None

        date = None
        if msg.get("Date"):
            try:
                date = parsedate_to_datetime(msg["Date"]).date().isoformat()
            except (TypeError, ValueError):
                date = None

        b = NodeBuilder()
        subject = header("Subject") or "(sans objet)"
        b.add("title", subject, page=1, level=1, method="plain")
        entete = " | ".join(
            f"{label} : {val}"
            for label, val in (
                ("De", header("From")),
                ("À", header("To")),
                ("Cc", header("Cc")),
                ("Date", date),
            )
            if val
        )
        chars = len(subject)
        if entete:
            b.add("paragraph", entete, page=1, method="plain", attributes={"email_header": True})
            chars += len(entete)

        body = msg.get_body(preferencelist=("plain", "html"))
        if body is not None:
            content = body.get_content()
            if body.get_content_type() == "text/html":
                chars += html_to_nodes(b, content)
            else:
                for block in re.split(r"\n\s*\n", content):
                    block = _clean(block)
                    if block:
                        chars += len(block)
                        b.add("paragraph", block, page=1, method="plain")

        attachments = [
            part.get_filename() for part in msg.iter_attachments() if part.get_filename()
        ]
        return ParsedDocument(
            file_name=path.name,
            source_path=str(path),
            doc_type="eml",
            metadata={
                "subject": header("Subject"),
                "from": header("From"),
                "to": header("To"),
                "date": date,
                "attachment_count": len(attachments),
                "attachments": ", ".join(attachments) or None,
            },
            pages=[
                PageInfo(
                    number=1,
                    method="plain" if chars else None,
                    quality=1.0 if chars else 0.0,
                    char_count=chars,
                )
            ],
            nodes=b.nodes(),
        )
