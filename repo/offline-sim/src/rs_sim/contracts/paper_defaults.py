from __future__ import annotations

"""Frozen execution semantics used by the paper-facing Current-P12 path.

These constants are shared by configuration, backend, scheduler and runtime so
omitting an option cannot silently select a different execution model.
"""

# ReleaseFrontier itself uses rank-local release.  The matched Local controls
# preserve the conventional phase barrier.  Keep both constants explicit so a
# paper experiment cannot silently compare two treatments under the same
# release semantics or, conversely, change only one treatment by accident.
PAPER_RELEASE_MODE = "RANK_LOCAL"
PAPER_LOCAL_RELEASE_MODE = "PHASE_BARRIER"
PAPER_JOINT_RELEASE_MODE = "RANK_LOCAL"
PAPER_RELEASE_ONLY_MODE = "RANK_LOCAL"
PAPER_P0_P1_COMPUTE_END_BARRIER = True
PAPER_MAX_TASK_BYTES = 256 * 1024
PAPER_ALIGNMENT_BYTES = 256


def require_paper_execution_semantics(
    *,
    p0_p1_compute_end_barrier: bool,
    max_task_bytes: int,
    alignment_bytes: int,
) -> None:
    """Validate execution settings shared by every paper treatment."""
    if p0_p1_compute_end_barrier is not PAPER_P0_P1_COMPUTE_END_BARRIER:
        raise ValueError(
            "paper execution requires the global pre-Combine P0->P1 compute-end barrier"
        )
    if int(max_task_bytes) != PAPER_MAX_TASK_BYTES:
        raise ValueError(
            f"paper execution requires max_task_bytes={PAPER_MAX_TASK_BYTES}"
        )
    if int(alignment_bytes) != PAPER_ALIGNMENT_BYTES:
        raise ValueError(
            f"paper execution requires alignment_bytes={PAPER_ALIGNMENT_BYTES}"
        )


def require_paper_treatment_release_semantics(
    *,
    scope: str,
    release_mode: str,
    experiment_role: str,
) -> None:
    """Validate the paper's Local / Joint semantics and explicit release ablations.

    Local baselines and the matched ReleaseFrontier-Local control retain the
    phase barrier.  The proposed Joint path uses rank-local release.  A
    phase-local ordering may use rank-local release only when it is explicitly
    labelled as an explicit Local rank-local-release ablation.
    """

    normalized_scope = str(scope).upper()
    normalized_release = str(release_mode).upper()
    normalized_role = str(experiment_role).upper()
    if normalized_scope == "WINDOW_JOINT":
        if normalized_release != PAPER_JOINT_RELEASE_MODE:
            raise ValueError(
                "paper Joint treatments require release_mode=RANK_LOCAL"
            )
        return
    if normalized_scope != "PHASE_LOCAL":
        raise ValueError(f"unsupported paper treatment scope {scope!r}")
    if normalized_role == "RELEASE_ONLY_ABLATION":
        if normalized_release != PAPER_RELEASE_ONLY_MODE:
            raise ValueError(
                "Local rank-local-release ablation requires release_mode=RANK_LOCAL"
            )
        return
    if normalized_release != PAPER_LOCAL_RELEASE_MODE:
        raise ValueError(
            "paper Local treatments require release_mode=PHASE_BARRIER; "
            "use experiment_role=RELEASE_ONLY_ABLATION for phase-local "
            "ordering with rank-local release"
        )
