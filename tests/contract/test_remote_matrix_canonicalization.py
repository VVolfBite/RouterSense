from __future__ import annotations

from rs.scheduling.traffic_matrix import (
    canonicalize_remote_matrix,
    matrix_col_sums_remote,
    matrix_diagonal_report,
    matrix_digest_remote,
    matrix_nonzero_remote_edge_count,
    matrix_remote_bytes,
    matrix_row_sums_remote,
    matrix_self_bytes,
    matrix_total_bytes,
)


def test_canonicalize_remote_matrix_zeros_diagonal() -> None:
    matrix = ((9, 4, -1), (3, 8, 5), (7, 6, 1))
    assert canonicalize_remote_matrix(matrix) == ((0, 4, 0), (3, 0, 5), (7, 6, 0))


def test_remote_byte_helpers_exclude_diagonal() -> None:
    matrix = ((9, 4, 0), (3, 8, 5), (7, 6, 1))
    assert matrix_total_bytes(matrix) == 43
    assert matrix_self_bytes(matrix) == 18
    assert matrix_remote_bytes(matrix) == 25
    assert matrix_nonzero_remote_edge_count(matrix) == 5
    assert matrix_row_sums_remote(matrix) == (4, 8, 13)
    assert matrix_col_sums_remote(matrix) == (10, 10, 5)


def test_remote_digest_ignores_diagonal_only_changes() -> None:
    clean = ((0, 4), (3, 0))
    dirty = ((999, 4), (3, 888))
    changed = ((0, 5), (3, 0))
    assert matrix_digest_remote(clean) == matrix_digest_remote(dirty)
    assert matrix_digest_remote(clean) != matrix_digest_remote(changed)


def test_matrix_diagonal_report_summarizes_self_traffic() -> None:
    report = matrix_diagonal_report(((10, 2), (3, 20)))
    assert report["total_bytes"] == 35
    assert report["remote_bytes"] == 5
    assert report["self_bytes"] == 30
    assert report["diagonal_nonzero_count"] == 2
