"""HTML to plain text.

selectolax is the fast path. The regex fallback exists so that the parse tests
-- and therefore CI on a machine without a C toolchain -- run without native
dependencies. Both paths must produce the same *shape* of output: block
elements become line breaks, inline elements do not.
"""

from __future__ import annotations

import html as html_module
import re

try:  # pragma: no cover - import guard
    from selectolax.parser import HTMLParser

    _HAVE_SELECTOLAX = True
except ImportError:  # pragma: no cover
    HTMLParser = None  # type: ignore[assignment]
    _HAVE_SELECTOLAX = False

#: Elements whose content is never readable text.
_DROP = ("script", "style", "head", "noscript", "template", "svg")

#: Elements that force a line break when flattened.
_BLOCK = {
    "address", "article", "blockquote", "br", "div", "dl", "dt", "dd", "fieldset",
    "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr",
    "li", "main", "nav", "ol", "p", "pre", "section", "table", "tbody", "td", "tfoot",
    "th", "thead", "tr", "ul",
}

_MULTI_NEWLINE = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")
_MULTI_SPACE = re.compile(r"[ \t ]{2,}")
_TAG = re.compile(r"<[^>]+>")
_DROP_RE = re.compile(
    r"<(script|style|head|noscript|template|svg)\b.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_BLOCK_RE = re.compile(r"</?(" + "|".join(sorted(_BLOCK)) + r")\b[^>]*>", re.IGNORECASE)


def html_to_text(html: str) -> str:
    """Flatten an HTML body to readable plain text."""
    if not html:
        return ""
    text = _with_selectolax(html) if _HAVE_SELECTOLAX else _with_regex(html)
    return tidy(text)


def _with_selectolax(html: str) -> str:
    tree = HTMLParser(html)
    for tag in _DROP:
        for node in tree.css(tag):
            node.decompose()

    root = tree.body or tree.root
    if root is None:
        return ""
    parts: list[str] = []
    _walk(root, parts, depth=0)
    return "".join(parts)


#: Marketing HTML nests tables absurdly deeply. Past this we stop recursing and
#: take the flat text of the subtree rather than risking a RecursionError.
_MAX_DEPTH = 200


def _walk(node, parts: list[str], depth: int) -> None:
    tag = node.tag
    if tag == "-text":
        txt = node.text(deep=False)
        if txt:
            parts.append(txt)
        return
    if tag in _DROP:
        return
    if depth >= _MAX_DEPTH:
        parts.append(node.text(deep=True) or "")
        return

    block = tag in _BLOCK
    if block:
        parts.append("\n")
    for child in node.iter(include_text=True):
        _walk(child, parts, depth + 1)
    if block:
        parts.append("\n")


def _with_regex(html: str) -> str:
    out = _DROP_RE.sub(" ", html)
    out = _BLOCK_RE.sub("\n", out)
    out = _TAG.sub(" ", out)
    return html_module.unescape(out)


def tidy(text: str) -> str:
    """Normalise whitespace without destroying paragraph structure.

    Paragraph breaks carry meaning for the model (a signature block, a quoted
    header) so blank lines are collapsed to one, never removed.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_SPACE.sub(" ", text)
    text = _TRAILING_SPACE.sub("\n", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()
