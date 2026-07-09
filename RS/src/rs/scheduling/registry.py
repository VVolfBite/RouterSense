from __future__ import annotations

from rs.scheduling.base import RouterSensePhasePolicy, SchedulingPolicy

from .phase_local.aurora_fixed import AuroraOrderFixedPolicy
from .phase_local.birkhoff_phase_local import BirkhoffPhaseLocalPolicy
from .phase_local.fast_bvn_fixed import FastBVNSingleTierPolicy
from .phase_local.fifo import BucketedFIFOPolicy, PhaseBarrierFIFOPolicy
from .phase_local.greedy_ready_set import GreedyReadySetPolicy
from .phase_local.islip_round_robin import ISLIPRoundRobinPolicy
from .phase_local.paired_family import BBarrierCriticalityMatchingPolicy, BBarrierPriceAdaptiveMatchingPolicy, BGatedGreedyMaximalPolicy, BGatedMaxweightMatchingPolicy, BLagrangianPhaseLocalPolicy
from .phase_local.power_of_two_choices import PowerOfTwoChoicesPolicy
from .phase_local.p0p1_reservation_order import RouterSenseP0P1ReservationPolicy
from .phase_local.p0p1p2_hint_order import RouterSenseP0P1P2HintPolicy
from .phase_local.trivial_reverse_bucket import TrivialReverseBucketPolicy
from .multiphase.safe_joint import SafeJointPolicy
from .runtime_bridge.joint_priority import RouterSenseJointPriorityPhaseSyncPolicy
from .multiphase.routersense_lookahead import RouterSenseMultiphaseLookaheadPolicy, UnsupportedOnlineMultiPhaseExecution
from .multiphase.recovered_candidates import is_recovered_candidate, resolve_recovered_candidate
from .multiphase.tier1 import TIER1_ALGORITHM_IDS, is_tier1_algorithm, resolve_tier1_policy
from .reference.birkhoff_von_neumann_fluid import BirkhoffVonNeumannFluidReference
from .reference.exact_small_instance import exact_result_to_logical_plan, solve_problem_exact


class NativePassthroughPolicy:
    policy_name = "native_passthrough"
    policy_version = "v1"
    capabilities = PhaseBarrierFIFOPolicy.capabilities.__class__(
        supports_offline=False,
        supports_online_phase_local_execution=False,
        supports_online_multiphase_execution=False,
        uses_current_ready_flows=False,
        uses_blocked_p1_dependency=False,
        uses_p2_forecast=False,
        requires_fixed_placement=False,
        evaluation_eligible=True,
    )

    def build_logical_plan(self, problem):
        del problem
        raise ValueError("native_passthrough does not build a logical schedule plan")


def _parse_policy_name(policy_name: str) -> tuple[str, str | None]:
    if policy_name.startswith("routersense_multiphase_lookahead:"):
        return "routersense_multiphase_lookahead", policy_name.split(":", 1)[1]
    return policy_name, None


def resolve_policy(
    *,
    policy_name: str,
    bucket_rows: int,
    p0_weight: float = 1.0,
    p1_reservation_weight: float = 1.0,
    p2_hint_weight: float = 1.0,
    p2_hint_artifact: str = "",
) -> SchedulingPolicy:
    base_name, mode = _parse_policy_name(policy_name)
    if base_name == "native_passthrough":
        return NativePassthroughPolicy()
    if base_name in {
        "RS_safe_gated_greedy",
        "RS_safe_gated_maxweight",
        "RS_safe_barrier_criticality",
        "RS_safe_barrier_price",
        "RS_safe_lagrangian",
        "RS_safe_ibbr",
    }:
        raw_u_name, paired_b_name = {
            "RS_safe_gated_greedy": ("U_gated_greedy_maximal", "B_gated_greedy_maximal"),
            "RS_safe_gated_maxweight": ("U_gated_maxweight_matching", "B_gated_maxweight_matching"),
            "RS_safe_barrier_criticality": ("U_barrier_criticality_global_matching", "B_barrier_criticality_matching"),
            "RS_safe_barrier_price": ("U_barrier_price_adaptive_matching", "B_barrier_price_adaptive_matching"),
            "RS_safe_lagrangian": ("U_lagrangian", "B_lagrangian_phase_local"),
            "RS_safe_ibbr": ("U_ibbr", "B_birkhoff"),
        }[base_name]
        raw_u_policy = resolve_policy(
            policy_name=raw_u_name,
            bucket_rows=bucket_rows,
            p0_weight=p0_weight,
            p1_reservation_weight=p1_reservation_weight,
            p2_hint_weight=p2_hint_weight,
            p2_hint_artifact=p2_hint_artifact,
        )
        paired_b_policy = resolve_policy(
            policy_name=paired_b_name,
            bucket_rows=bucket_rows,
            p0_weight=p0_weight,
            p1_reservation_weight=p1_reservation_weight,
            p2_hint_weight=p2_hint_weight,
            p2_hint_artifact=p2_hint_artifact,
        )
        return SafeJointPolicy(
            policy_name=base_name,
            raw_u_policy_name=raw_u_name,
            paired_b_policy_name=paired_b_name,
            raw_u_policy=raw_u_policy,
            paired_b_policy=paired_b_policy,
        )
    if is_tier1_algorithm(base_name):
        return resolve_tier1_policy(base_name)
    if is_recovered_candidate(base_name):
        return resolve_recovered_candidate(base_name)
    if base_name == "phase_barrier_fifo":
        return PhaseBarrierFIFOPolicy(bucket_rows=bucket_rows, reported_policy_name=base_name)
    if base_name == "bucketed_fifo":
        return BucketedFIFOPolicy(bucket_rows=bucket_rows)
    if base_name == "greedy_ready_set":
        return GreedyReadySetPolicy(bucket_rows=bucket_rows)
    if base_name == "islip_round_robin":
        return ISLIPRoundRobinPolicy(bucket_rows=bucket_rows)
    if base_name == "power_of_two_choices":
        return PowerOfTwoChoicesPolicy(bucket_rows=bucket_rows)
    if base_name == "birkhoff_phase_local":
        return BirkhoffPhaseLocalPolicy(bucket_rows=bucket_rows)
    if base_name == "B_gated_greedy_maximal":
        return BGatedGreedyMaximalPolicy(bucket_rows=bucket_rows)
    if base_name == "B_gated_maxweight_matching":
        return BGatedMaxweightMatchingPolicy(bucket_rows=bucket_rows)
    if base_name == "B_barrier_criticality_matching":
        return BBarrierCriticalityMatchingPolicy(bucket_rows=bucket_rows)
    if base_name == "B_barrier_price_adaptive_matching":
        return BBarrierPriceAdaptiveMatchingPolicy(bucket_rows=bucket_rows)
    if base_name == "B_lagrangian_phase_local":
        return BLagrangianPhaseLocalPolicy(bucket_rows=bucket_rows)
    if base_name == "birkhoff_von_neumann_fluid":
        return BirkhoffVonNeumannFluidReference(bucket_rows=bucket_rows)
    if base_name == "exact_small_instance_reference":
        return ExactSmallInstanceReferencePolicy()
    if base_name == "trivial_reverse_bucket":
        return TrivialReverseBucketPolicy(bucket_rows=bucket_rows)
    if base_name == "aurora_order_fixed":
        return AuroraOrderFixedPolicy(bucket_rows=bucket_rows)
    if base_name == "fast_bvn_single_tier":
        return FastBVNSingleTierPolicy(bucket_rows=bucket_rows)
    if base_name == "routersense_p0p1_reservation":
        return RouterSenseP0P1ReservationPolicy(
            bucket_rows=bucket_rows,
            p0_weight=p0_weight,
            p1_reservation_weight=p1_reservation_weight,
        )
    if base_name == "routersense_p0p1p2_hint":
        return RouterSenseP0P1P2HintPolicy(
            bucket_rows=bucket_rows,
            p0_weight=p0_weight,
            p1_reservation_weight=p1_reservation_weight,
            p2_hint_weight=p2_hint_weight,
        )
    if base_name == "routersense_joint_priority_phase_sync":
        return RouterSenseJointPriorityPhaseSyncPolicy(
            bucket_rows=bucket_rows,
            p0_weight=p0_weight,
            p1_reservation_weight=p1_reservation_weight,
            p2_hint_weight=p2_hint_weight,
        )
    if base_name == "routersense_multiphase_lookahead":
        return RouterSenseMultiphaseLookaheadPolicy(
            information_mode=mode or "p0_p1_p2",
            p0_weight=p0_weight,
            p1_reservation_weight=p1_reservation_weight,
            p2_hint_weight=p2_hint_weight,
        )
    raise ValueError(f"Unknown policy {policy_name!r}")


def resolve_phase_policy(
    *,
    policy_name: str,
    bucket_rows: int,
    p0_weight: float = 1.0,
    p1_reservation_weight: float = 1.0,
    p2_hint_weight: float = 1.0,
    p2_hint_artifact: str = "",
) -> RouterSensePhasePolicy:
    policy = resolve_policy(
        policy_name=policy_name,
        bucket_rows=bucket_rows,
        p0_weight=p0_weight,
        p1_reservation_weight=p1_reservation_weight,
        p2_hint_weight=p2_hint_weight,
        p2_hint_artifact=p2_hint_artifact,
    )
    if not getattr(policy.capabilities, "supports_online_phase_local_execution", False):
        raise UnsupportedOnlineMultiPhaseExecution(
            f"{policy_name!r} requires multiphase_pending_window and cannot run on the current phase-local online executor"
        )
    return policy


def supported_phase_policies() -> tuple[str, ...]:
    return (
        "phase_barrier_fifo",
        "bucketed_fifo",
        "greedy_ready_set",
        "islip_round_robin",
        "power_of_two_choices",
        "birkhoff_phase_local",
        "trivial_reverse_bucket",
        "aurora_order_fixed",
        "fast_bvn_single_tier",
        "routersense_p0p1_reservation",
        "routersense_p0p1p2_hint",
        "routersense_joint_priority_phase_sync",
    )


def supported_policies() -> tuple[str, ...]:
    return (
        "native_passthrough",
        "phase_barrier_fifo",
        "bucketed_fifo",
        "greedy_ready_set",
        "islip_round_robin",
        "power_of_two_choices",
        "birkhoff_phase_local",
        "B_gated_greedy_maximal",
        "B_gated_maxweight_matching",
        "B_barrier_criticality_matching",
        "B_barrier_price_adaptive_matching",
        "B_lagrangian_phase_local",
        "RS_safe_gated_greedy",
        "RS_safe_gated_maxweight",
        "RS_safe_barrier_criticality",
        "RS_safe_barrier_price",
        "RS_safe_lagrangian",
        "RS_safe_ibbr",
        "birkhoff_von_neumann_fluid",
        "exact_small_instance_reference",
        "trivial_reverse_bucket",
        "aurora_order_fixed",
        "fast_bvn_single_tier",
        "routersense_p0p1_reservation",
        "routersense_p0p1p2_hint",
        "routersense_joint_priority_phase_sync",
        "routersense_multiphase_lookahead:p0_only",
        "routersense_multiphase_lookahead:p0_p1",
        "routersense_multiphase_lookahead:p0_p1_p2",
        "U_gated_greedy_maximal",
        "U_gated_greedy_maximal_atomic",
        "U_ibbr",
        "U_barrier_price_adaptive_matching",
        "U_barrier_price_adaptive_matching_atomic",
        "RS_safe_gated_greedy",
        "RS_safe_gated_maxweight",
        "RS_safe_barrier_criticality",
        "RS_safe_barrier_price",
        "RS_safe_lagrangian",
        "RS_safe_ibbr",
        *TIER1_ALGORITHM_IDS,
    )


class ExactSmallInstanceReferencePolicy:
    policy_name = "exact_small_instance_reference"
    policy_version = "v1"
    capabilities = PhaseBarrierFIFOPolicy.capabilities.__class__(
        supports_offline=True,
        supports_online_phase_local_execution=False,
        supports_online_multiphase_execution=False,
        uses_current_ready_flows=True,
        uses_blocked_p1_dependency=False,
        uses_p2_forecast=False,
        requires_fixed_placement=True,
        evaluation_eligible=True,
    )

    def build_logical_plan(self, problem):
        result = solve_problem_exact(problem)
        if not result.get("supported", False):
            return exact_result_to_logical_plan(result, policy_name=self.policy_name)
        return exact_result_to_logical_plan(result, policy_name=self.policy_name)
