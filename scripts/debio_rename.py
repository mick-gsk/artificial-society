#!/usr/bin/env python3
"""Token-accurate identifier de-biologizer (dev tool; not shipped in the package).

Rewrites Python *identifier* word-parts per a concept map, leaving string
literals and comments untouched. Word-part aware: a NAME token is split on
underscores, each segment further split on CamelCase / digit boundaries; only
parts that equal a map key (case-insensitive) are replaced, each preserving its
original case pattern. So `traits`, `derive_traits`, `GeneStore` are renamed
while `generation`, `genesis`, `generate`, `eigener` are not.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import tokenize
from pathlib import Path

# ALLCAPS run | Titlecase word | lowercase run | digit run
_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+")


def _recase(replacement: str, sample: str) -> str:
    """Cast `replacement` to the case pattern of the original word `sample`."""
    if sample.isupper() and not sample.islower():
        return replacement.upper()
    if sample[:1].isupper() and sample[1:].islower():
        return replacement[:1].upper() + replacement[1:]
    return replacement  # lowercase / mixed -> verbatim (map values are snake/lower)


def rename_identifier(name: str, mapping: dict[str, str]) -> str:
    """Return `name` with every matching word-part replaced; layout preserved."""
    segments = re.split(r"(_+)", name)  # keep underscore separators
    out: list[str] = []
    for seg in segments:
        if not seg or seg.startswith("_"):
            out.append(seg)  # separator or empty (leading/trailing/dunder)
            continue
        words = _WORD.findall(seg)
        if not words or "".join(words) != seg:
            out.append(seg)  # non-standard token: leave untouched
            continue
        rebuilt = [_recase(mapping[w.lower()], w) if w.lower() in mapping else w for w in words]
        out.append("".join(rebuilt))
    return "".join(out)


def rewrite_source(src: str, mapping: dict[str, str]):
    """Rewrite NAME tokens in `src`; return (new_src, {"old->new": count})."""
    lines = src.splitlines(keepends=True)
    edits: dict[int, list[tuple[int, int, str]]] = {}
    changes: dict[str, int] = {}
    readline = iter(lines).__next__
    for tok in tokenize.generate_tokens(readline):
        if tok.type != tokenize.NAME:
            continue
        new = rename_identifier(tok.string, mapping)
        if new == tok.string:
            continue
        (srow, scol), (erow, ecol) = tok.start, tok.end
        if srow != erow:
            continue  # NAME tokens are single-line
        edits.setdefault(srow, []).append((scol, ecol, new))
        key = f"{tok.string}->{new}"
        changes[key] = changes.get(key, 0) + 1
    for row, row_edits in edits.items():
        line = lines[row - 1]
        for scol, ecol, new in sorted(row_edits, reverse=True):
            line = line[:scol] + new + line[ecol:]
        lines[row - 1] = line
    return "".join(lines), changes


def _sub_words(text: str, mapping: dict[str, str]) -> str:
    def repl(m):
        w = m.group(0)
        r = mapping.get(w.lower())
        return _recase(r, w) if r is not None else w

    return re.sub(r"[A-Za-z]+", repl, text)


def rewrite_prose(src: str, mapping: dict[str, str]) -> str:
    """Rewrite words only inside COMMENT tokens and docstrings; code strings kept."""
    mapping = {k.lower(): v for k, v in mapping.items()}
    # docstring line ranges (module/class/func first-statement string expressions)
    doc_ranges = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(body, list) and body:
                first = body[0]
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(getattr(first, "value", None), ast.Constant)
                    and isinstance(first.value.value, str)
                ):
                    for ln in range(first.lineno, first.end_lineno + 1):
                        doc_ranges.add(ln)
    lines = src.splitlines(keepends=True)
    readline = iter(lines).__next__
    edits: dict[int, list[tuple[int, int, str]]] = {}
    for tok in tokenize.generate_tokens(readline):
        if tok.type == tokenize.COMMENT and tok.start[0] == tok.end[0]:
            new = _sub_words(tok.string, mapping)
            if new != tok.string:
                edits.setdefault(tok.start[0], []).append((tok.start[1], tok.end[1], new))
    for row in sorted(doc_ranges):
        line = lines[row - 1]
        edits.setdefault(row, []).append(
            (0, len(line.rstrip("\n")), _sub_words(line.rstrip("\n"), mapping))
        )
    for row, row_edits in edits.items():
        line = lines[row - 1]
        for scol, ecol, new in sorted(row_edits, reverse=True):
            line = line[:scol] + new + line[ecol:]
        lines[row - 1] = line
    return "".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, help="JSON file: {part: replacement}")
    ap.add_argument("--apply", action="store_true", help="write files (default: dry-run)")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args(argv)
    mapping = {k.lower(): v for k, v in json.loads(Path(args.map).read_text()).items()}
    total: dict[str, int] = {}
    for f in args.files:
        p = Path(f)
        try:
            src = p.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        new, changes = rewrite_source(src, mapping)
        if changes:
            for k, v in changes.items():
                total[k] = total.get(k, 0) + v
            if args.apply:
                p.write_text(new)
    for k in sorted(total):
        print(f"{total[k]:5d}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
