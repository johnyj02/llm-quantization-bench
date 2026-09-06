#!/usr/bin/env python3
"""Gate G6's desk-reject checks that are not page counts or section prose.

    python scripts/check_submittable.py paper/main.tex paper/main.pdf

🚨 **WHY EACH CHECK EXISTS IS WRITTEN NEXT TO IT, and that is the point of this file**
(planner ruling R42.2). G6's instrument has now drifted from G6's intent twice:

1. It compared TOTAL PDF pages against 8, when ACL's limit applies to the body only. That
   over-counted the moment Limitations existed and nearly caused body content to be cut to
   make room for content that never competed for the space.
2. Its Limitations check was `grep -q 'section\\*{Limitations}'`, which tested that the
   *heading* existed. It reported PASS while five required sections carried zero prose, and
   an empty Limitations section is a desk reject exactly as a missing one is.

That is this project's recurring *declared versus enforced* failure arriving one level up, in
the instrument that checks the deliverable. The structural fix, per R42.2, is that **every
check names the desk-reject ground it protects, inline**, so the next reader can see whether
the instrument still tests the intent instead of discovering it by failing.

⚠️ **Every ground below was verified against the ARR CFP on 2026-09-05**, not from recall,
after the planner declined to be the authority on a desk-reject criterion. Two beliefs the
planner and the executor shared turned out to be unsupported: that Acknowledgments is
excluded from the page count (the CFP does not address it), and that a ninth page is granted
on acceptance (the CFP states no such thing).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Author and affiliation strings that must never appear in an anonymous submission.
#: Deliberately drawn from the repo's own git identity rather than typed, so it cannot go
#: stale against a rename.
def _identity_terms() -> list[str]:
    terms: list[str] = []
    for key in ("user.name", "user.email"):
        try:
            value = subprocess.run(
                ["git", "config", "--get", key], cwd=REPO,
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
        if value:
            terms.extend(part for part in re.split(r"[\s@.]+", value) if len(part) > 3)
    return sorted({t.lower() for t in terms})


def check_anonymity(pdf_text: str) -> list[str]:
    """DESK-REJECT GROUND: a non-anonymous submission.

    ARR review is anonymous, and a name, institution or resolvable artifact URL in the PDF
    breaks it. Trivially checkable and previously unchecked. Also catches the S3 bucket and
    account id, which appear throughout this repo's operational notes and would identify the
    authors' organisation immediately.
    """
    failures = []
    lowered = pdf_text.lower()
    for term in _identity_terms():
        if term in lowered:
            failures.append(f"identity term {term!r} appears in the PDF")
    # Resolvable locations that would deanonymise or leak infrastructure.
    for pattern, what in (
        (r"github\.com/[\w.-]+/[\w.-]+", "a GitHub repository URL"),
        (r"s3://[\w.-]+", "an S3 URI"),
        (r"\b\d{12}\b", "a 12-digit AWS account id"),
        (r"[\w.-]+\.amazonaws\.com", "an AWS hostname"),
        (r"dkr\.ecr\.", "an ECR image reference"),
    ):
        for m in re.finditer(pattern, pdf_text):
            failures.append(f"{what}: {m.group(0)!r}")
    return failures


def check_references_resolve(pdf_text: str) -> list[str]:
    """DESK-REJECT-ADJACENT: silently dropped citations and unresolved references.

    `CLAUDE.md` records this as a defect this project has already had: BibTeX has no
    top-level comment character, so an `@` inside what looks like a comment starts an entry
    and a `%` inside an entry is parsed as a field name. **Both drop citations silently while
    the PDF still builds.**

    ⚠️ Checked against the PDF, never against `main.log`. On 2026-09-05 grepping the log for
    "undefined" returned ZERO while the PDF contained `??` from four broken cross-references.
    The log is not a reliable witness; the rendered document is.
    """
    failures = []
    if "??" in pdf_text:
        n = pdf_text.count("??")
        failures.append(f"{n} unresolved reference(s) rendered as '??' in the PDF")
    # A citation that failed to resolve renders as a bracketed query in most styles.
    if re.search(r"\[\?\]", pdf_text):
        failures.append("unresolved citation rendered as '[?]' in the PDF")
    return failures


def check_todo_deadline(tex: str, today: str) -> list[str]:
    """SUBMITTABILITY: a reader must never see a red TODO marker.

    G6 permits `\\todo{}` inside Results because numbers arrive late, and that is right while
    numbers are still arriving. It stops being right at the point the draft leaves our hands.

    🚨 Planner ruling R42.2 sets the deadline at the **external-reader handoff (Sep 30)**,
    not the results freeze: numbers stop arriving at G7 (Sep 25), and anything still marked
    by Sep 30 is either a number that will never exist, in which case cut the claim, or prose
    that needs writing. "Always submittable" and "todos allowed indefinitely" genuinely
    conflict after that date, and this is where the conflict is resolved.
    """
    # ⚠️ Comments must be stripped PER LINE, not pattern-matched. A first version counted
    # `re.findall(r"\\todo\{")` minus occurrences of the literal `"% \\todo{"`, which
    # matched only one of the three comment forms in this file and reported 6 where 4 were
    # real. A gate that miscounts is the drift this module exists to prevent, so the comment
    # boundary is found properly: everything after an unescaped `%` is a LaTeX comment.
    n = 0
    for line in tex.splitlines():
        code = re.split(r"(?<!\\)%", line, maxsplit=1)[0]
        n += code.count("\\todo{")
    if n <= 0:
        return []
    if today >= "2026-09-30":
        return [f"{n} \\todo{{}} marker(s) remain, and the Sep 30 reader handoff has passed"]
    print(f"  note     {n} \\todo{{}} marker(s) remain; permitted until 2026-09-30 (R42.2)")
    return []


def check_checklist(_: str) -> list[str]:
    """DESK-REJECT GROUND: an incorrect or incomplete Responsible NLP Checklist.

    `A23` records that an incomplete or incorrect filing has itself been a desk-reject ground
    since Dec 2024, and G6's original four criteria did not cover it at all. Same failure
    class as the Limitations grep: a desk-reject ground with no instrument.

    It is not greppable from the PDF, so the artifact is a file with every item answered and
    no blanks. §C (compute, hyperparameters, error bars) and §E (AI assistance) apply
    directly to us.

    ⚠️ Filling §D will force a statement about whether the human annotation actually ran. If
    it did not, the judge rung is unvalidated and §3's gate was never executed. The checklist
    is exactly where that inconsistency surfaces, and better here than at review.
    """
    path = REPO / "paper" / "responsible_nlp_checklist.md"
    if not path.is_file():
        return [f"missing {path.relative_to(REPO)} (A23: incomplete filing is a desk reject)"]
    text = path.read_text()
    blanks = [
        line.strip() for line in text.splitlines()
        if re.match(r"^\s*[-*]\s*\S.*:\s*$", line) or "TODO" in line or "TBD" in line
    ]
    return [f"checklist has {len(blanks)} unanswered item(s), first: {blanks[0]!r}"] if blanks else []


def main() -> int:
    tex_path = Path(sys.argv[1] if len(sys.argv) > 1 else "paper/main.tex")
    pdf_path = Path(sys.argv[2] if len(sys.argv) > 2 else "paper/main.pdf")
    today = subprocess.run(["date", "-u", "+%Y-%m-%d"], capture_output=True,
                           text=True).stdout.strip()

    tex = tex_path.read_text()
    try:
        pdf_text = subprocess.run(
            ["pdftotext", str(pdf_path), "-"], capture_output=True, text=True, timeout=120,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        # ⚠️ Fails LOUD. A check that cannot run must not report success: that is the
        # `except: return {}` shape that silently restored an earlier bug in this repo.
        print(f"FAIL: could not extract PDF text ({exc}). Install poppler (pdftotext).")
        return 1

    checks = {
        "anonymity": check_anonymity(pdf_text),
        "references resolve": check_references_resolve(pdf_text),
        "todo deadline": check_todo_deadline(tex, today),
        "Responsible NLP Checklist": check_checklist(tex),
    }

    failed = 0
    for name, problems in checks.items():
        if problems:
            failed += 1
            print(f"  FAIL     {name}")
            for p in problems[:6]:
                print(f"           - {p}")
            if len(problems) > 6:
                print(f"           - ... and {len(problems) - 6} more")
        else:
            print(f"  ok       {name}")

    if failed:
        print(f"\nFAIL: {failed} of {len(checks)} desk-reject checks failed.")
        return 1
    print("\nPASS: all desk-reject checks clear.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
