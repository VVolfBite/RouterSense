from __future__ import annotations

"""Canonical phase identity helpers for Current-P12 windows."""

from rs_sim.contracts.schema import PhaseKey, PhaseKind


def next_layer_dispatch_phase(current_phase: PhaseKey) -> PhaseKey:
    """Return the single authority key shared by P2_l and P0_{l+1}."""

    if not isinstance(current_phase, PhaseKey):
        raise TypeError("current_phase must be PhaseKey")
    return PhaseKey(
        run_id=current_phase.run_id,
        sample_id=current_phase.sample_id,
        layer_index=int(current_phase.layer_index) + 1,
        phase_kind=PhaseKind.DISPATCH,
    )


def assert_same_dispatch_authority(left: PhaseKey, right: PhaseKey) -> None:
    if not isinstance(left, PhaseKey) or not isinstance(right, PhaseKey):
        raise TypeError("phase aliases must be PhaseKey")
    if left != right or left.phase_kind is not PhaseKind.DISPATCH:
        raise ValueError("P2_l and P0_{l+1} must resolve to the same Dispatch PhaseKey")


__all__ = ["assert_same_dispatch_authority", "next_layer_dispatch_phase"]
