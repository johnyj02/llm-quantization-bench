"""Analysis layer: the statistics `analysis/preregistration.md` registers.

Task `A19` in `EXECUTOR-TASKS.md` §6, and the pre-registered sections it implements:

| module | registers |
|---|---|
| `results` | §9's reporting rules: effect sizes with CIs for everything, never a bare significance flag, Holm-Bonferroni within a family, and §9f's three invocation-noise classes |
| `stratum` | §1.1's always-compliant stratum, as amended by R18.4.1 (three contrasted arms), §9g (all invocations) and R24.3 (threshold dissolved) |
| `transitions` | §1.2's compliance-transition matrix with exact McNemar |
| `bounds` | §1.3's Manski bounds on the full-sample competence effect |
| `interaction` | §2's H3 precision x scoring interaction, as a marginal effect and an odds ratio |
| `decompose` | H4' from §4, four components, and §4's pre-specified `TRUNC_STOPSEQ` differentiality test |

Nothing here touches a GPU or reads a generation. Every function consumes scoring rows over
stored generations, which is the same property that makes the artifact release useful: a third
party can re-run the whole analysis without an accelerator.

⚠️ One deliberate omission. There is no `verdict()` anywhere in this package. §2 pre-commits
that *"a null interaction is a publishable, confirmatory result"*, §1.3 that wide bounds are an
honest result, and §4 that H1 stays the lead whatever `TRUNC_STOPSEQ` turns out to be. Those are
readings that belong in the paper beside an estimate and its interval, so the code returns
estimates and intervals and declines to rank them.
"""

from .bounds import ArmBounds, BoundsError, DifferenceBounds, bounds_effects, compute_bounds
from .decompose import (
    EXCLUDED_FROM_H4_PRIMARY,
    Decomposition,
    DecompositionError,
    decompose,
    stopseq_differential,
)
from .interaction import (
    EXCLUDED_FROM_H3_INTERACTION,
    HEADLINE_RUNGS,
    HETEROGENEITY_CONTRASTS,
    InteractionError,
    heterogeneity_contrast,
    interaction_glmm,
    interaction_marginal,
    rung_gap_decomposition,
    rung_mcnemar,
    study_wide_magnitude,
)
from .results import Effect, NoiseClass, holm_bonferroni
from .stratum import (
    CONTRASTED_ARMS,
    StratumError,
    StratumResult,
    compute_stratum,
)
from .transitions import (
    TransitionError,
    TransitionTable,
    build_transitions,
    mcnemar_exact,
    transition_effects,
)

__all__ = [
    "CONTRASTED_ARMS", "EXCLUDED_FROM_H3_INTERACTION", "EXCLUDED_FROM_H4_PRIMARY",
    "HEADLINE_RUNGS",
    "ArmBounds", "BoundsError", "Decomposition", "DecompositionError",
    "DifferenceBounds", "Effect", "InteractionError", "NoiseClass",
    "StratumError", "StratumResult", "TransitionError", "TransitionTable",
    "bounds_effects", "build_transitions", "compute_bounds", "compute_stratum",
    "decompose", "holm_bonferroni", "interaction_glmm", "HETEROGENEITY_CONTRASTS",
    "heterogeneity_contrast",
    "interaction_marginal",
    "mcnemar_exact", "rung_gap_decomposition", "rung_mcnemar", "stopseq_differential", "study_wide_magnitude",
    "transition_effects",
]
