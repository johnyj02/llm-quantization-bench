#!/usr/bin/env python3
"""Fail if a required paper section exists but is EMPTY. Part of gate G6.

    python scripts/check_sections.py paper/main.tex

🚨 WHY THIS EXISTS. G6's Limitations check was `grep -q 'section\\*{Limitations}'` — it
tested that the *heading* exists. On 2026-08-31 an audit found the Introduction,
Discussion, Conclusion, Limitations, Ethics Statement and all three appendices carried
**zero prose**: heading, a label, some `%` planning comments, nothing else. G6 reported
`PASS` on all four of its criteria.

A missing Limitations section is an **automatic desk reject** (`EXECUTOR-TASKS.md` A23).
An empty one is the same thing wearing a heading, and the gate could not tell the
difference. It is also the project's own recurring failure mode — declared versus
enforced — arriving at the level of the instrument that checks the deliverable.

Comments do not count as content, deliberately: `% write this later` is a plan, not a
Limitations section. Neither does a bare `\\todo{}`, except in Results, where G6 permits
todos until the results freeze.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Sections that must carry real prose for the paper to be submittable. Appendices are
# excluded: they are permitted to be stubs, and A23 requires they carry no load-bearing
# content anyway.
REQUIRED = (
    "Introduction",
    "Related Work",
    "Method",
    "Results",
    "Discussion",
    "Conclusion",
    "Limitations",
    "Ethics Statement",
)

# Structural markup is not prose. A section containing only a table is not written.
_STRUCTURAL = re.compile(
    r"\\(label|todo|input|begin|end|centering|small|footnotesize|caption|"
    r"toprule|midrule|bottomrule|cmidrule|newcommand|multicolumn|hline|"
    r"section|subsection|paragraph|"
    # ⚠️ Counter machinery is markup, not prose. The `end-of-body` marker that G6's page
    # count reads sits between Conclusion and Limitations, so it falls inside Conclusion's
    # span; without these two, an EMPTY Conclusion reported "ok (2 prose lines)" and the
    # desk-reject guard gave a false pass on the section it exists to protect.
    r"newcounter|refstepcounter|phantomsection|setcounter|addtocounter)\b"
)

# `\section{X}` and `\section*{X}`, plus subsections so a parent section counts the
# prose of its children — Method's own body is empty but its subsections are written,
# and that is legitimately a written Method.
_HEADING = re.compile(r"\\(section\*?|subsection\*?)\{(.+?)\}")


def prose_lines(lines: list[str]) -> int:
    n = 0
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        if _STRUCTURAL.match(line):
            continue
        n += 1
    return n


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "main.tex")
    lines = path.read_text().splitlines()

    headings: list[tuple[int, str, bool]] = []
    for i, line in enumerate(lines):
        match = _HEADING.search(line)
        if match:
            headings.append((i, match.group(2), match.group(1).startswith("section")))
    headings.append((len(lines), "\x00EOF", True))

    # A top-level section spans until the next top-level section, so its span includes
    # its subsections' prose.
    spans: dict[str, int] = {}
    for index, (start, name, is_section) in enumerate(headings[:-1]):
        if not is_section:
            continue
        end = len(lines)
        for later_start, _, later_is_section in headings[index + 1:]:
            if later_is_section:
                end = later_start
                break
        spans[name] = prose_lines(lines[start + 1:end])

    empty = [name for name in REQUIRED if spans.get(name, 0) == 0]

    for name in REQUIRED:
        count = spans.get(name)
        if count is None:
            print(f"  MISSING  {name}")
        else:
            mark = "EMPTY  " if count == 0 else "ok     "
            print(f"  {mark}  {name}  ({count} prose lines)")

    missing = [name for name in REQUIRED if name not in spans]
    if missing:
        print(f"\nFAIL: section heading absent: {', '.join(missing)}")
        return 1
    if empty:
        print(f"\nFAIL: {len(empty)} required section(s) have a heading but NO prose: "
              f"{', '.join(empty)}")
        if "Limitations" in empty:
            print("      🚨 An empty Limitations section is a desk reject exactly as a "
                  "missing one is (A23).")
        return 1
    print("\nPASS: every required section carries prose.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
