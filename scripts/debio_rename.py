#!/usr/bin/env python3
"""Token-accurate identifier de-biologizer (dev tool; not shipped in the package).

Rewrites Python *identifier* word-parts per a concept map, leaving string
literals and comments untouched. Word-part aware: a NAME token is split on
underscores, each segment further split on CamelCase / digit boundaries; only
parts that equal a map key (case-insensitive) are replaced, each preserving its
original case pattern. So `genes`, `inherit_genes`, `GeneStore` are renamed
while `generation`, `genesis`, `generate`, `eigener` are not.
"""

from __future__ import annotations

import argparse
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
