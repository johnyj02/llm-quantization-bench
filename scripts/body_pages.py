#!/usr/bin/env python3
"""Pages of MAIN BODY, which is what ACL's 8-page limit actually applies to.

    python scripts/body_pages.py paper/main.aux

🚨 **Why this is not `pdf_pages.py`.** ACL counts Limitations, Ethics, Acknowledgments,
references and appendices as **zero page budget** (`EXECUTOR-TASKS.md` A23 states it for
Limitations explicitly). Gate G6 was comparing the TOTAL PDF page count against 8, which
over-counts as soon as any of those sections exists.

That is not a cosmetic error. On 2026-09-05, writing Limitations took the total to exactly 8
and I began treating the page budget as binding, which would have meant cutting body content
to make room for content that does not compete with it. A gate that measures the wrong
quantity does not merely fail to catch problems, it manufactures them.

The body ends at `\\label{end-of-body}` in `main.tex`, placed immediately before Limitations.
LaTeX writes that label's page number into `main.aux`, so this reads the number LaTeX itself
assigned rather than inferring it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# \newlabel{end-of-body}{{}{7}{}{}{}} -- the SECOND brace group is the page.
_LABEL = re.compile(r"\\newlabel\{end-of-body\}\{\{[^}]*\}\{(\d+)\}")


def body_pages(aux: Path) -> int:
    match = _LABEL.search(aux.read_text())
    if match is None:
        raise SystemExit(
            f"{aux}: no \\newlabel{{end-of-body}}. main.tex must carry "
            f"\\label{{end-of-body}} immediately before \\section*{{Limitations}}, and the "
            f"document must have been compiled at least twice so the label resolves."
        )
    return int(match.group(1))


if __name__ == "__main__":
    print(body_pages(Path(sys.argv[1] if len(sys.argv) > 1 else "main.aux")))
