from __future__ import annotations


def test_p0_collective_pair_contract_counts_hidden_probs_and_p1() -> None:
    rows = [
        {"tensor_role": "p0_hidden_collective"},
        {"tensor_role": "p0_probs_collective"},
        {"tensor_role": "p1_hidden_collective"},
    ]
    assert sum(1 for row in rows if row["tensor_role"] == "p0_hidden_collective") == 1
    assert sum(1 for row in rows if row["tensor_role"] == "p0_probs_collective") == 1
    assert sum(1 for row in rows if row["tensor_role"] == "p1_hidden_collective") == 1
