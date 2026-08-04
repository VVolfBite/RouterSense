"""Converters from explicit token-level post-policy routing observations.

The converter consumes the kept/drop decision as captured truth.  It does not
apply an expert-capacity policy itself.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from ..schema.model import RealizedRouting, TraceValidationError


def realized_routing_from_token_assignments(
    *,
    selected_experts_by_source: Sequence[Sequence[Sequence[int]]],
    kept_mask_by_source: Sequence[Sequence[Sequence[bool]]],
    num_experts: int,
    padding_rows_by_source_expert: Iterable[Iterable[int]] | None = None,
    realization_origin: str = "captured_token_assignments_with_explicit_keep_mask",
) -> RealizedRouting:
    """Aggregate captured top-k assignments and explicit keep/drop outcomes.

    Shapes:
      selected_experts_by_source[source][token][choice]
      kept_mask_by_source[source][token][choice]

    A `True` mask entry contributes to kept rows; `False` contributes to dropped
    rows. Padding is supplied independently because it is an explicit transfer
    realization, not a token routing decision.
    """
    if int(num_experts) <= 0:
        raise TraceValidationError("num_experts must be positive")
    if len(selected_experts_by_source) != len(kept_mask_by_source):
        raise TraceValidationError("selected/kept source dimensions differ")
    world_size = len(selected_experts_by_source)
    if world_size <= 0:
        raise TraceValidationError("at least one source rank is required")
    raw = [[0 for _ in range(num_experts)] for _ in range(world_size)]
    kept = [[0 for _ in range(num_experts)] for _ in range(world_size)]
    dropped = [[0 for _ in range(num_experts)] for _ in range(world_size)]
    for source_rank, (source_tokens, source_masks) in enumerate(zip(selected_experts_by_source, kept_mask_by_source)):
        if len(source_tokens) != len(source_masks):
            raise TraceValidationError(f"source {source_rank}: token dimensions differ")
        for token_index, (choices, masks) in enumerate(zip(source_tokens, source_masks)):
            if len(choices) != len(masks):
                raise TraceValidationError(f"source {source_rank} token {token_index}: top-k dimensions differ")
            for expert_id, is_kept in zip(choices, masks):
                expert = int(expert_id)
                if expert < 0 or expert >= int(num_experts):
                    raise TraceValidationError(
                        f"source {source_rank} token {token_index}: expert_id={expert} outside num_experts"
                    )
                raw[source_rank][expert] += 1
                if bool(is_kept):
                    kept[source_rank][expert] += 1
                else:
                    dropped[source_rank][expert] += 1
    if padding_rows_by_source_expert is None:
        padding = [[0 for _ in range(num_experts)] for _ in range(world_size)]
    else:
        padding = [[int(v) for v in row] for row in padding_rows_by_source_expert]
        if len(padding) != world_size or any(len(row) != int(num_experts) for row in padding):
            raise TraceValidationError("padding matrix shape must be [world_size][num_experts]")
    return RealizedRouting.from_lists(
        raw_selected_rows=raw,
        kept_rows=kept,
        dropped_rows=dropped,
        padding_rows=padding,
        realization_origin=realization_origin,
    )
