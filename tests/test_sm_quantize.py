"""Tests for the `quantize` entry point's REPLICATE path (job 1b, planner ruling R18.3).

Recipe-level tests live in `test_quantize_recipe.py`; this file covers only the plumbing
that job 1b added, because that plumbing failed on its first run in a way the project has
now seen five times.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

SM_QUANTIZE = (REPO / "sm_quantize.py").read_text()


class TestReplicateDestinationReachesTheChild:
    """🚨 REGRESSION, job 1b attempt 1 (2026-08-30, 277 s, $0.22).

    The parent suffixed `destination` with `-rep2` while `quantize_model` recomputed its
    own path from `out_root`. The child therefore found the ORIGINAL study checkpoint
    already present, returned it, and the parent then failed looking for a directory that
    was never written -- having done no work at all.

    Nothing was damaged: `quantize_model`'s idempotence guard refused to overwrite job 1's
    artifact, so a resume convenience turned out to be a safety net. But this is the
    project's most repeated failure shape -- **a value set in one process and silently not
    honoured in another** (`desc_act`, `async_scheduling`, `add_bos_token`, the pilot's
    `until`, and now this). The resolved path must travel explicitly; these tests pin the
    whole chain so it cannot be half-reverted.
    """

    def test_payload_carries_dest_override(self):
        assert '"dest_override": str(destination) if REPLICATE else None' in SM_QUANTIZE

    def test_child_forwards_dest_override_to_quantize_model(self):
        assert 'dest_override=(Path(payload["dest_override"])' in SM_QUANTIZE

    def test_quantize_model_honours_the_override(self):
        from quantartifact.quantize import quantize_model  # noqa: PLC0415

        assert "dest_override" in inspect.signature(quantize_model).parameters
        assert "dest_override or output_dir(" in inspect.getsource(quantize_model), (
            "the override must take precedence over the derived path"
        )

    def test_replicate_suffixes_the_name_and_never_the_digest(self):
        """The digest stays in the directory name: it is the artifact's own proof that
        both builds used an identical recipe, verifiable without trusting us."""
        assert "destination.with_name(destination.name + REPLICATE_SUFFIX)" in SM_QUANTIZE

    def test_replicate_refuses_to_combine_with_smoke(self):
        """The smoke uses a deliberately wrong 8x256 calibration, so a replicate built
        under it would compare nothing while looking like a pass."""
        assert "mutually exclusive" in SM_QUANTIZE

    def test_replicate_never_resumes_a_cached_replicate(self):
        """The question is whether a FRESH build reproduces; a cached one answers
        nothing. Same reasoning as the smoke discarding prior artifacts."""
        assert "discarded a previous replicate" in SM_QUANTIZE

    def test_replicate_targets_exactly_one_arm(self):
        """Explicit rather than derived, so this can never rebuild the whole matrix."""
        assert 'REPLICATE_TARGET = ("qwen2.5-3b", "awq-4bit")' in SM_QUANTIZE

    def test_rmtree_is_guarded_by_a_containment_assert(self):
        """Deleting a path built by string concatenation warrants a guard that it is
        inside the checkpoint dir before any rmtree."""
        assert SM_QUANTIZE.count("assert CHECKPOINT_DIR in destination.parents") >= 2
