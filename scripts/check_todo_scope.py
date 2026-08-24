#!/usr/bin/env python3
"""Gate G6: assert \\todo{} markers appear ONLY inside the Results section.

EXECUTOR-TASKS.md 0.1, "Rule of the always-submittable artifact": from Sep 20
there must exist a compiling paper/main.pdf at <=8 pages with every section
written, with \\todo{} permitted ONLY inside Results. This is the single
highest-value insurance policy in the plan, so it is checked mechanically
rather than by eye.

Comment lines are ignored -- a commented-out \\todo{} is a note to ourselves,
not an unwritten section.
"""

from __future__ import annotations

import re
import sys

# \todo appearing inside these sections is allowed.
ALLOWED_SECTIONS = {"Results"}

SECTION_RE = re.compile(r"\\section\*?\{([^}]*)\}")
# The abstract is a section that must be written too; report it by name rather
# than as "<preamble>", which reads like a checker bug.
ABSTRACT_BEGIN_RE = re.compile(r"\\begin\{abstract\}")
ABSTRACT_END_RE = re.compile(r"\\end\{abstract\}")
TODO_RE = re.compile(r"\\todo\{")
# The \newcommand that DEFINES \todo is not a use of it.
DEFINITION_RE = re.compile(r"\\newcommand\{?\\todo")


def main(path: str) -> int:
    current = "<preamble>"
    violations: list[tuple[int, str, str]] = []
    allowed_count = 0

    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()

    for lineno, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")

        stripped = line.lstrip()
        if stripped.startswith("%"):
            continue
        # Strip trailing comments, honouring escaped \%.
        line = re.sub(r"(?<!\\)%.*$", "", line)

        if ABSTRACT_BEGIN_RE.search(line):
            current = "Abstract"
        elif ABSTRACT_END_RE.search(line):
            current = "<preamble>"

        section = SECTION_RE.search(line)
        if section:
            current = section.group(1).strip()

        if DEFINITION_RE.search(line):
            continue

        for _ in TODO_RE.finditer(line):
            if current in ALLOWED_SECTIONS:
                allowed_count += 1
            else:
                violations.append((lineno, current, line.strip()))

    if violations:
        print(f"  FAIL: {len(violations)} \\todo{{}} outside Results:")
        for lineno, section, text in violations:
            print(f"    {path}:{lineno}  [in '{section}']  {text[:70]}")
        return 1

    print(f"  PASS ({allowed_count} \\todo{{}} in Results, 0 elsewhere)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "main.tex"))
