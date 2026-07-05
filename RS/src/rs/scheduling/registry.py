from __future__ import annotations

from rs.scheduling.base import RouterSensePhasePolicy

from .phase_local.aurora_fixed import AuroraOrderFixedPolicy
from .phase_local.fast_bvn_fixed import FastBVNSingleTierPolicy
from .phase_local.fifo import BucketedFIFOPolicy
from .phase_local.p0p1_reservation_order import RouterSenseP0P1ReservationPolicy
from .phase_local.p0p1p2_hint_order import RouterSenseP0P1P2HintPolicy
from .phase_local.trivial_reverse_bucket import TrivialReverseBucketPolicy


def resolve_phase_policy(
    *,
    policy_name: str,
    bucket_rows: int,
    p0_weight: float = 1.0,
    p1_reservation_weight: float = 1.0,
    p2_hint_weight: float = 1.0,
    p2_hint_artifact: str = "",
) -> RouterSensePhasePolicy:
    if policy_name == "bucketed_fifo":
        return BucketedFIFOPolicy(bucket_rows=bucket_rows)
    if policy_name == "trivial_reverse_bucket":
        return TrivialReverseBucketPolicy(bucket_rows=bucket_rows)
    if policy_name == "aurora_order_fixed":
        return AuroraOrderFixedPolicy(bucket_rows=bucket_rows)
    if policy_name == "fast_bvn_single_tier":
        return FastBVNSingleTierPolicy(bucket_rows=bucket_rows)
    if policy_name == "routersense_p0p1_reservation":
        return RouterSenseP0P1ReservationPolicy(
            bucket_rows=bucket_rows,
            p0_weight=p0_weight,
            p1_reservation_weight=p1_reservation_weight,
        )
    if policy_name == "routersense_p0p1p2_hint":
        return RouterSenseP0P1P2HintPolicy(
            bucket_rows=bucket_rows,
            p0_weight=p0_weight,
            p1_reservation_weight=p1_reservation_weight,
            p2_hint_weight=p2_hint_weight,
        )
    raise ValueError(f"Unknown phase policy {policy_name!r}")


def supported_phase_policies() -> tuple[str, ...]:
    return (
        "bucketed_fifo",
        "trivial_reverse_bucket",
        "aurora_order_fixed",
        "fast_bvn_single_tier",
        "routersense_p0p1_reservation",
        "routersense_p0p1p2_hint",
    )
