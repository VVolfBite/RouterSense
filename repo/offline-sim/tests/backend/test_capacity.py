from rs_sim.backend import compute_staging_capacity_bytes


def test_staging_sensitivity_uses_exact_rational_and_max_task_floor():
    assert compute_staging_capacity_bytes(
        receiver_buffer_reference_bytes=256,
        sensitivity="0.25x",
        alignment_bytes=16,
        max_canonical_task_payload_bytes=64,
    ) == 64
    assert compute_staging_capacity_bytes(
        receiver_buffer_reference_bytes=250,
        sensitivity="0.5x",
        alignment_bytes=16,
        max_canonical_task_payload_bytes=32,
    ) == 128
    assert compute_staging_capacity_bytes(
        receiver_buffer_reference_bytes=256,
        sensitivity="UNBOUNDED",
        alignment_bytes=16,
        max_canonical_task_payload_bytes=64,
    ) is None


class _CapacityWindow:
    def payload_matrix(self, phase_kind):
        del phase_kind
        return (
            (900, 40),
            (40, 900),
        )


class _CapacityFixture:
    world_size = 2
    windows = (_CapacityWindow(),)


def test_fixture_capacity_excludes_diagonal_and_uses_actual_remote_edge_floor():
    from rs_sim.backend import compute_fixture_staging_capacity_bytes_by_rank

    capacities = compute_fixture_staging_capacity_bytes_by_rank(
        fixture_input=_CapacityFixture(),
        sensitivity="0.25X",
        alignment_bytes=1,
        max_canonical_task_payload_bytes=256,
    )
    # Remote inbound is 40 bytes; diagonal 900-byte local assembly must not
    # inflate the pool, and the configured 256-byte upper bound is not a floor.
    assert capacities == {0: 40, 1: 40}
