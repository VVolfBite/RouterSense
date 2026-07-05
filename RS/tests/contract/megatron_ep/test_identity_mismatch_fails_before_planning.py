from __future__ import annotations

import pytest

from rs.scheduling.validation import validate_global_observations

from .helpers import make_context, make_observation


def test_identity_mismatch_fails_before_planning() -> None:
    context = make_context()
    left = make_observation(rank=0, phase="P0", rows=(0, 5), placement_hash="placement-a")
    right = make_observation(rank=1, phase="P0", rows=(4, 0), placement_hash="placement-b")
    with pytest.raises(ValueError, match="expert_placement_hash"):
        validate_global_observations(context, (left, right))
