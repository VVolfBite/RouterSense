"""Phase-local scheduling policies with lazy import."""

from __future__ import annotations

__all__ = [
    "AuroraOrderFixedPolicy",
    "BirkhoffPhaseLocalPolicy",
    "BucketedFIFOPolicy",
    "FastBVNSingleTierPolicy",
    "GreedyReadySetPolicy",
    "ISLIPRoundRobinPolicy",
    "PhaseBarrierFIFOPolicy",
    "RouterSenseP0P1P2HintPolicy",
    "RouterSenseP0P1ReservationPolicy",
    "TrivialReverseBucketPolicy",
]


def __getattr__(name: str):
    if name == "AuroraOrderFixedPolicy":
        from .aurora_fixed import AuroraOrderFixedPolicy

        return AuroraOrderFixedPolicy
    if name == "BirkhoffPhaseLocalPolicy":
        from .birkhoff_phase_local import BirkhoffPhaseLocalPolicy

        return BirkhoffPhaseLocalPolicy
    if name == "FastBVNSingleTierPolicy":
        from .fast_bvn_fixed import FastBVNSingleTierPolicy

        return FastBVNSingleTierPolicy
    if name in {"BucketedFIFOPolicy", "PhaseBarrierFIFOPolicy"}:
        from .fifo import BucketedFIFOPolicy, PhaseBarrierFIFOPolicy

        return {
            "BucketedFIFOPolicy": BucketedFIFOPolicy,
            "PhaseBarrierFIFOPolicy": PhaseBarrierFIFOPolicy,
        }[name]
    if name == "GreedyReadySetPolicy":
        from .greedy_ready_set import GreedyReadySetPolicy

        return GreedyReadySetPolicy
    if name == "ISLIPRoundRobinPolicy":
        from .islip_round_robin import ISLIPRoundRobinPolicy

        return ISLIPRoundRobinPolicy
    if name == "RouterSenseP0P1ReservationPolicy":
        from .p0p1_reservation_order import RouterSenseP0P1ReservationPolicy

        return RouterSenseP0P1ReservationPolicy
    if name == "RouterSenseP0P1P2HintPolicy":
        from .p0p1p2_hint_order import RouterSenseP0P1P2HintPolicy

        return RouterSenseP0P1P2HintPolicy
    if name == "TrivialReverseBucketPolicy":
        from .trivial_reverse_bucket import TrivialReverseBucketPolicy

        return TrivialReverseBucketPolicy
    raise AttributeError(name)
