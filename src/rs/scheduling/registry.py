"""Resolver for formal phase-local controls and offline reference baselines."""
from __future__ import annotations

from rs.reference.baselines import (
    AuroraOrderFixedPolicy,
    FastBVNSingleTierPolicy,
    GMWDStylePolicy,
    ISLIPRoundRobinPolicy,
    PowerOfTwoChoicesPolicy,
    TrivialReverseBucketPolicy,
)
from rs.scheduling.catalog import algorithm_specs, phase_local_algorithm_specs, resolve_algorithm_id
from rs.scheduling.phase_local.birkhoff_phase_local import BirkhoffPhaseLocalPolicy
from rs.scheduling.phase_local.fifo import BucketedFIFOPolicy
from rs.scheduling.phase_local.greedy_ready_set import GreedyReadySetPolicy
from rs.scheduling.reference.birkhoff_von_neumann_fluid import BirkhoffVonNeumannFluidReference


def resolve_policy(*, policy_name: str, bucket_rows: int = 0, **_: object):
    spec = resolve_algorithm_id(policy_name).spec
    name = spec.canonical_id
    if name == "fifo_bucket":
        return BucketedFIFOPolicy(bucket_rows=bucket_rows)
    if name == "greedy_bucket":
        return GreedyReadySetPolicy(bucket_rows=bucket_rows)
    if name == "birkhoff_bucket_phase_local":
        return BirkhoffPhaseLocalPolicy(bucket_rows=bucket_rows)
    if name == "gmwd_style_reference":
        return GMWDStylePolicy(bucket_rows=bucket_rows)
    if name == "islip_reference":
        return ISLIPRoundRobinPolicy(bucket_rows=bucket_rows)
    if name == "fast_stage_reference":
        return FastBVNSingleTierPolicy(bucket_rows=bucket_rows)
    if name == "aurora_order_reference":
        return AuroraOrderFixedPolicy(bucket_rows=bucket_rows)
    if name == "power_of_two_reference":
        return PowerOfTwoChoicesPolicy(bucket_rows=bucket_rows)
    if name == "reverse_bucket_reference":
        return TrivialReverseBucketPolicy(bucket_rows=bucket_rows)
    if name == "birkhoff_fluid_reference":
        return BirkhoffVonNeumannFluidReference(bucket_rows=bucket_rows)
    if spec.execution_model == "exact_reference":
        raise ValueError("exact algorithms must be invoked through OracleRegistry")
    raise ValueError(f"unknown formal policy {policy_name!r}")


def resolve_phase_policy(*, policy_name: str, bucket_rows: int, **kwargs: object):
    resolved = resolve_algorithm_id(policy_name)
    if not resolved.spec.phase_local_eligible:
        raise ValueError(f"{policy_name!r} is not a deployable phase-local policy")
    return resolve_policy(policy_name=resolved.canonical_name, bucket_rows=bucket_rows, **kwargs)


def supported_phase_policies() -> tuple[str, ...]:
    return tuple(spec.canonical_id for spec in phase_local_algorithm_specs())


def supported_policies() -> tuple[str, ...]:
    return tuple(spec.canonical_id for spec in algorithm_specs())


__all__ = ["resolve_phase_policy", "resolve_policy", "supported_phase_policies", "supported_policies"]
