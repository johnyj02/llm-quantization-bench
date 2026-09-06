#!/usr/bin/env python3
"""Pages of MAIN BODY, which is what ACL's 8-page limit actually applies to.

    python scripts/body_pages.py paper/main.aux

🚨 **Why this is not `pdf_pages.py`.** The ARR CFP excludes Limitations, Ethics,
references and appendices from the page count, so gate G6 comparing the TOTAL PDF page count
against 8 over-counts as soon as any of those sections exists.

Verified against the ARR CFP directly on 2026-09-05 rather than from recall, and the
verification corrected this docstring twice:

- **Limitations:** *"included at the end of the paper, before the references, and it will
  not count toward the page limit"*, and it is **required**.
- **Ethics:** excluded *"as long as it is placed in the end of the paper"*.
- **References:** *"unlimited pages of references"*.
- **Appendices:** *"They do not count towards the page limit"*, but they must use the
  official double-column format or the submission is desk-rejected (July 2025 onward), and
  putting key content there *"could be viewed as an attempt to circumvent the content page
  count"*, which is itself a desk-reject risk.
- ⚠️ **Acknowledgments: NOT addressed by the CFP.** This docstring previously asserted it
  was excluded, and that claim was unsupported. It does not bite for us because ours is
  empty and sits after Ethics, and an anonymous submission should not carry one at all, but
  the assertion was wrong and is removed rather than left as a plausible guess.
- ⚠️ **There is no stated extra page on acceptance.** A widely repeated belief, which the
  CFP does not support: it gives only the 8-page long and 4-page short submission limits and
  draws no camera-ready distinction. Do not plan against a ninth page.

**Exceeding the limit is a desk rejection**, and the CFP notes desk rejections *"can happen
at any time in the review cycle"*, not only at the initial check.

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
