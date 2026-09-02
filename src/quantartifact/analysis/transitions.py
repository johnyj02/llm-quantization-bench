"""The compliance-transition matrix: §1.2 of the pre-registration.

§1.2 is emphatic that this is *"a substantive result"* rather than a nuisance to be conditioned
away: *"Compliance shift is a real quantization effect and belongs in the paper. Prior work
folds it into 'reasoning accuracy'; we report it separately."*

For each **ordered** precision pair `(bf16 -> p_alt)` the registered output is a 2x2 table:

|                          | committed under `p_alt` | abstained under `p_alt` |
|--------------------------|-------------------------|-------------------------|
| committed under `bf16`   | n11                     | n12                     |
| abstained under `bf16`   | n21                     | n22                     |

reported per (model x task x rung), with the off-diagonal asymmetry `n12 - n21` and an
**exact** McNemar test on the discordant cells.

Two implementation points that matter for correctness rather than style:

* **Exact, not the chi-square approximation.** §1.2 says exact, and the discordant counts here
  can be small: the approximation is unreliable below roughly 25 discordant pairs and this
  study has cells with a handful. `statsmodels.stats.contingency_tables.mcnemar(exact=True)`
  runs a two-sided binomial test on `n12` out of `n12 + n21`, which is the right test.
* **The pairing is per item, so both arms must have a record for an item or it is excluded and
  counted.** A paired test over unpaired data is a wrong number that looks plausible, and the
  count of dropped items is reported so the exclusion is visible rather than inferred.

⚠️ This quantity contrasts two arms, so it is **noise class 3** under §9f: two independent
invocations, variances add. The floor is not applied here (it is measured from the K=2 cells in
`main-v1` and attached by the caller), but the class is recorded on every effect so it cannot
be compared against the wrong floor later.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .results import Effect, NoiseClass
from .stratum import ABSENT_LABEL, COMMITTED_LABELS

#: §1.2 fixes bf16 as the reference and ranges `p_alt` over the four quantized arms.
REFERENCE_ARM = "bf16"
ALT_ARMS: tuple[str, ...] = ("gptq-8bit", "gptq-4bit", "awq-4bit", "bnb-nf4")


class TransitionError(RuntimeError):
    """Raised when a transition table cannot be built, rather than built wrongly."""


@dataclass(frozen=True)
class TransitionTable:
    """One 2x2 compliance-transition table for one (cell, rung, arm pair)."""

    cell: tuple
    rung: str
    reference_arm: str
    alt_arm: str
    #: committed under both.
    n11: int
    #: committed under reference, abstained under alt. A compliance LOSS from quantizing.
    n12: int
    #: abstained under reference, committed under alt. A compliance GAIN.
    n21: int
    #: abstained under both.
    n22: int
    #: Items seen for this cell that could not be paired, and are therefore excluded.
    n_unpaired: int

    @property
    def n(self) -> int:
        return self.n11 + self.n12 + self.n21 + self.n22

    @property
    def n_discordant(self) -> int:
        return self.n12 + self.n21

    @property
    def asymmetry(self) -> int:
        """`n12 - n21`, the registered summary. Positive means quantizing lost commitments."""
        return self.n12 - self.n21

    def as_dict(self) -> dict:
        return {
            "cell": list(self.cell), "rung": self.rung,
            "reference_arm": self.reference_arm, "alt_arm": self.alt_arm,
            "n11": self.n11, "n12": self.n12, "n21": self.n21, "n22": self.n22,
            "n": self.n, "n_discordant": self.n_discordant, "asymmetry": self.asymmetry,
            "n_unpaired": self.n_unpaired,
        }


def _is_committed(label: str) -> bool:
    if label in COMMITTED_LABELS:
        return True
    if label == ABSENT_LABEL:
        return False
    raise TransitionError(
        f"unrecognised compliance label {label!r}. The 2x2 is defined on committed versus "
        f"abstained only, so an unknown third label cannot be placed in it. Extend "
        f"COMMITTED_LABELS deliberately rather than letting this fall through."
    )


def mcnemar_exact(n12: int, n21: int) -> tuple[float | None, str]:
    """Two-sided exact McNemar on the discordant cells.

    Returns `(p_value, test_name)`, with `p_value` `None` when there are no discordant pairs.
    That is not a p-value of 1.0: with zero discordant pairs the test has no data and saying
    so is different from saying the arms agreed perfectly by chance. The caller decides how to
    report it, and `Effect` refuses a p-value without a named test anyway.
    """
    if n12 < 0 or n21 < 0:
        raise ValueError(f"negative discordant counts: n12={n12}, n21={n21}")
    if n12 + n21 == 0:
        return None, "mcnemar_exact"
    from statsmodels.stats.contingency_tables import mcnemar  # noqa: PLC0415

    # `exact=True` runs the binomial test; `correction` is irrelevant in that branch. The table
    # is laid out so that mcnemar() reads the off-diagonal as (n12, n21).
    result = mcnemar([[0, n12], [n21, 0]], exact=True)
    return float(result.pvalue), "mcnemar_exact"


def build_transitions(
    rows: list[dict],
    *,
    rungs: tuple[str, ...],
    reference_arm: str = REFERENCE_ARM,
    alt_arms: tuple[str, ...] = ALT_ARMS,
) -> list[TransitionTable]:
    """Build every registered transition table from per-(item, arm) scoring rows.

    Rows follow the same shape `stratum.compute_stratum` consumes. Where a cell has more than
    one invocation per arm (the K=2 cells), the **first** invocation is used and the choice is
    recorded, because §1.2's table is defined on a single pair of measurements per item and
    averaging two invocations into a binary commitment label is not defined.
    """
    if not rungs:
        raise ValueError("build_transitions needs at least one rung")

    # cell -> rung -> arm -> item -> committed, taking repeat_idx 0 only.
    index: dict[tuple, dict[str, dict[str, dict[str, bool]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict)))
    for row in rows:
        if row.get("repeat_idx", 0) != 0:
            continue
        cell = (row["model_key"], row["task"], row.get("subtask"),
                row.get("budget_level", "1x"), row.get("prompt_variant", "fewshot_cot"))
        for rung in rungs:
            if rung not in row["compliance"]:
                raise TransitionError(
                    f"{cell} item {row['item_id']} arm {row['arm']}: no compliance label for "
                    f"rung {rung!r}"
                )
            index[cell][rung][row["arm"]][row["item_id"]] = _is_committed(
                row["compliance"][rung])

    tables: list[TransitionTable] = []
    for cell, by_rung in sorted(index.items(), key=lambda kv: str(kv[0])):
        for rung in rungs:
            by_arm = by_rung[rung]
            if reference_arm not in by_arm:
                raise TransitionError(
                    f"{cell} rung {rung}: reference arm {reference_arm!r} has no rows, so no "
                    f"transition table is defined. §1.2 fixes bf16 as the reference."
                )
            ref = by_arm[reference_arm]
            for alt_arm in alt_arms:
                if alt_arm not in by_arm:
                    continue  # arm not present in this condition, e.g. the 3-arm sweep cells
                alt = by_arm[alt_arm]
                shared = set(ref) & set(alt)
                counts = {"n11": 0, "n12": 0, "n21": 0, "n22": 0}
                for item_id in shared:
                    r, a = ref[item_id], alt[item_id]
                    key = "n11" if (r and a) else "n12" if (r and not a) else \
                          "n21" if (not r and a) else "n22"
                    counts[key] += 1
                tables.append(TransitionTable(
                    cell=cell, rung=rung,
                    reference_arm=reference_arm, alt_arm=alt_arm,
                    n_unpaired=len(set(ref) ^ set(alt)),
                    **counts,
                ))
    return tables


def transition_effects(tables: list[TransitionTable]) -> list[Effect]:
    """Turn each table into an `Effect` on the compliance-shift rate.

    The estimate is the **net compliance shift in percentage points**, `(n21 - n12) / n * 100`,
    signed so that a negative value means quantizing cost commitments. Reporting it as a rate
    rather than as the raw asymmetry makes it comparable across cells of different size, and the
    raw counts stay on the table for the appendix.

    The interval is the Wilson interval on the discordant proportion, rescaled to the rate. It
    is the interval that pairs with the exact McNemar test rather than a normal approximation.
    """
    from statsmodels.stats.proportion import proportion_confint  # noqa: PLC0415

    effects: list[Effect] = []
    for table in tables:
        if table.n == 0:
            # An empty table is reported as such by omission from the effects list; it must not
            # become an effect of zero with a zero-width interval, which would read as a
            # precise null. The table itself is still returned by build_transitions.
            continue
        net_pp = (table.n21 - table.n12) / table.n * 100.0
        p_value, test = mcnemar_exact(table.n12, table.n21)

        if table.n_discordant == 0:
            # No discordant pairs: the shift is exactly zero and the only honest interval is
            # the Wilson interval on 0 out of n, which is not zero-width.
            low, high = proportion_confint(0, table.n, alpha=0.05, method="wilson")
            ci_low, ci_high = -float(high) * 100.0, float(high) * 100.0
        else:
            # Interval on the split of the discordant pairs, mapped onto the net rate.
            low, high = proportion_confint(table.n21, table.n_discordant,
                                           alpha=0.05, method="wilson")
            scale = table.n_discordant / table.n * 100.0
            ci_low = (2.0 * float(low) - 1.0) * scale
            ci_high = (2.0 * float(high) - 1.0) * scale

        effects.append(Effect(
            name=f"compliance_shift[{table.alt_arm} vs {table.reference_arm}]",
            estimate=net_pp, ci_low=ci_low, ci_high=ci_high,
            unit="pp", n=table.n, p_value=p_value, test=test,
            noise_class=NoiseClass.COMPOUNDING,
            context={"cell": list(table.cell), "rung": table.rung,
                     "n11": table.n11, "n12": table.n12,
                     "n21": table.n21, "n22": table.n22,
                     "n_unpaired": table.n_unpaired},
        ))
    return effects
