"""The always-compliant stratum: §1.1 of the pre-registration, amended by §9g and R24.3.

This is the load-bearing identification section. `EXECUTOR-PLAN.md` §4.0 sets out the problem
it solves: competence is `P(correct | committed)`, but **whether the model commits is itself
affected by precision**, so comparing competence across arms conditions on a post-treatment
variable. That is collider bias, and §4.0 notes the bias points *toward* this paper's own
hypothesis, which is why it is fixed by construction rather than argued away.

The fix is to compute the primary competence contrast only on items where **every contrasted
arm committed**, which makes the subgroup pre-treatment-defined with respect to any single
precision contrast.

Three amendments are registered and implemented here, and each changes membership:

* **R18.4.1** narrows the conjunction to the **three contrasted arms** (`bf16`, `gptq-4bit`,
  `awq-4bit`) rather than all five. Requiring commitment from arms we do not contrast shrinks
  the stratum multiplicatively for no identification benefit.
* **§9g** requires membership over **all invocations of all contrasted arms**, not over a
  K=2-averaged commitment, because invocation noise can flip commitment and the stricter
  version is the one that cannot be accused of being invocation-shopped. Both sizes are
  reported so the cost of the stricter choice is visible rather than hidden.
* **R24.3** dissolves the 50% minimum threshold. It was proportional where it should have been
  absolute (27% of 1,319 GSM8K items is 356 items, while 60% of a 150-item BBH subtask is 90),
  so the rule ranked them the wrong way round. The point estimate **and** the Manski bounds are
  now reported for every cell, always, with the stratum size as a first-class number, and which
  of the two is primary becomes a presentation choice rather than an analysis gate.

⚠️ One trap from §1.1 that is implemented explicitly below because getting it backwards would
shrink the stratum for no inferential gain: **`committed` and `TRUNC_BUDGET` are not mutually
exclusive.** A generation reading *"The answer is 42. Let me verify: 42 x 2 = "* is both
committed and budget-truncated. Truncation **after** commitment does not bias the competence
contrast, because the model already told us its answer, so those items belong in the stratum.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

# §1.1 as amended by R18.4.1. Order is fixed so `bf16` is always the reference arm.
CONTRASTED_ARMS: tuple[str, ...] = ("bf16", "gptq-4bit", "awq-4bit")

# §1.1: committed means the item produced an answer we can score, which includes the case
# where the answer was only recoverable from the trace. `ABSENT` is the sole non-committed
# label. Imported as literals rather than from `scoring.compliance` to keep this module
# importable without the scorer, and asserted against it in the tests.
COMMITTED_LABELS: frozenset[str] = frozenset({"COMPLIANT", "RECOVERABLE"})
ABSENT_LABEL = "ABSENT"


class StratumError(RuntimeError):
    """Raised when membership cannot be determined, rather than guessed.

    A missing arm is not an empty stratum. If a contrasted arm produced no record for an item,
    the conjunction over arms is undefined for that item, and returning `False` would quietly
    treat a data gap as a model abstention. That conflation of "I cannot know this" with "the
    model did not commit" is the exact shape of the `classify_termination` bug recorded in
    `notes/reviews/2026-09-01T020424Z.md`.
    """


@dataclass(frozen=True)
class StratumResult:
    """Membership and both registered sizes for one cell."""

    #: The cell this describes: (model_key, task, subtask, budget_level, prompt_variant).
    cell: tuple
    #: Item ids in the stratum under §9g, i.e. committed in ALL invocations of all arms.
    item_ids: frozenset[str]
    #: Item ids under the laxer alternative §9g rejected: committed in at least one
    #: invocation per arm. Reported so the cost of the stricter rule is visible.
    item_ids_any_invocation: frozenset[str]
    #: Every item id seen for this cell, i.e. the denominator.
    n_items_total: int
    #: Per-arm count of items that were ABSENT in at least one invocation, which is what
    #: excluded them. Names which arm is doing the shrinking.
    excluded_by_arm: dict[str, int]
    #: Number of invocations seen per arm, so a K=2 cell is visibly K=2.
    invocations_by_arm: dict[str, int]
    rungs: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.item_ids)

    @property
    def fraction(self) -> float | None:
        """Stratum size as a fraction of items. `None` when the cell has no items.

        Returns `None` rather than 0.0 for an empty cell, because a cell with no data and a
        cell where every item was excluded are different findings.
        """
        if self.n_items_total == 0:
            return None
        return self.size / self.n_items_total

    @property
    def fraction_any_invocation(self) -> float | None:
        if self.n_items_total == 0:
            return None
        return len(self.item_ids_any_invocation) / self.n_items_total

    @property
    def invocation_cost(self) -> int:
        """How many items the stricter §9g rule excluded that the laxer one would keep.

        This is the number §9g asks to be reported: the price of the invocation-stable
        definition, in items.
        """
        return len(self.item_ids_any_invocation) - self.size

    def as_dict(self) -> dict:
        return {
            "cell": list(self.cell),
            "size": self.size,
            "fraction": self.fraction,
            "size_any_invocation": len(self.item_ids_any_invocation),
            "fraction_any_invocation": self.fraction_any_invocation,
            "invocation_cost_items": self.invocation_cost,
            "n_items_total": self.n_items_total,
            "excluded_by_arm": dict(self.excluded_by_arm),
            "invocations_by_arm": dict(self.invocations_by_arm),
            "rungs": list(self.rungs),
        }


def _cell_of(row: dict) -> tuple:
    return (row["model_key"], row["task"], row.get("subtask"),
            row.get("budget_level", "1x"), row.get("prompt_variant", "fewshot_cot"))


def compute_stratum(
    rows: list[dict],
    *,
    rungs: tuple[str, ...],
    contrasted_arms: tuple[str, ...] = CONTRASTED_ARMS,
) -> dict[tuple, StratumResult]:
    """Compute the always-compliant stratum per cell.

    `rows` are per-(item, arm, invocation) scoring rows. Each must carry `model_key`, `task`,
    `subtask`, `item_id`, `arm`, `repeat_idx`, and a `compliance` mapping from rung name to a
    compliance label. `budget_level` and `prompt_variant` default to the reference condition.

    Membership, per §1.1: item `i` is in the stratum for its cell iff for every arm `p` in
    `contrasted_arms` and every rung `s` in `rungs`, `compliance(i, p, s)` is in
    `COMMITTED_LABELS`. Per §9g the quantifier over arms extends over **every invocation** of
    each arm.

    Raises `StratumError` if a cell is missing an entire contrasted arm, because the
    conjunction is then undefined rather than false.
    """
    if not rungs:
        raise ValueError(
            "compute_stratum needs at least one rung. An empty rung tuple makes the "
            "conjunction over rungs vacuously true, so every item would enter the stratum "
            "regardless of compliance."
        )

    # cell -> item -> arm -> list of per-invocation committed booleans
    seen: dict[tuple, dict[str, dict[str, list[bool]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    invocations: dict[tuple, dict[str, set]] = defaultdict(lambda: defaultdict(set))

    for row in rows:
        arm = row["arm"]
        if arm not in contrasted_arms:
            continue  # non-contrasted arms are reported elsewhere, not part of this conjunction
        cell = _cell_of(row)
        compliance = row["compliance"]
        missing = [r for r in rungs if r not in compliance]
        if missing:
            raise StratumError(
                f"{cell} item {row['item_id']} arm {arm}: no compliance label for rung(s) "
                f"{missing}. Membership is a conjunction over rungs, so a missing rung cannot "
                f"be treated as either committed or absent."
            )
        committed = all(compliance[r] in COMMITTED_LABELS for r in rungs)
        seen[cell][row["item_id"]][arm].append(committed)
        invocations[cell][arm].add(row.get("repeat_idx", 0))

    results: dict[tuple, StratumResult] = {}
    for cell, items in seen.items():
        present_arms = set(invocations[cell])
        absent_arms = set(contrasted_arms) - present_arms
        if absent_arms:
            raise StratumError(
                f"{cell}: contrasted arm(s) {sorted(absent_arms)} produced no rows. The "
                f"stratum is a conjunction across {list(contrasted_arms)}, so it is undefined "
                f"here rather than empty. Score the missing arm or exclude the cell explicitly."
            )

        strict_members: set[str] = set()
        lax_members: set[str] = set()
        excluded_by_arm: dict[str, int] = {arm: 0 for arm in contrasted_arms}

        for item_id, by_arm in items.items():
            if set(by_arm) != set(contrasted_arms):
                # An item missing an arm entirely is excluded and counted, not an error: with
                # nested subsamples a cell can legitimately hold items only some arms ran.
                for arm in set(contrasted_arms) - set(by_arm):
                    excluded_by_arm[arm] += 1
                continue
            # §9g: committed in ALL invocations of ALL arms.
            all_invocations = all(all(flags) for flags in by_arm.values())
            # The alternative §9g rejected: committed in at least one invocation of each arm.
            any_invocation = all(any(flags) for flags in by_arm.values())
            if all_invocations:
                strict_members.add(item_id)
            if any_invocation:
                lax_members.add(item_id)
            if not all_invocations:
                for arm, flags in by_arm.items():
                    if not all(flags):
                        excluded_by_arm[arm] += 1

        results[cell] = StratumResult(
            cell=cell,
            item_ids=frozenset(strict_members),
            item_ids_any_invocation=frozenset(lax_members),
            n_items_total=len(items),
            excluded_by_arm=excluded_by_arm,
            invocations_by_arm={a: len(v) for a, v in invocations[cell].items()},
            rungs=tuple(rungs),
        )
    return results
