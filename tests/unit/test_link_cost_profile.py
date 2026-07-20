from __future__ import annotations

import json

import pytest

from rs.topology.link_cost import fit_affine_row_cost, load_link_cost_profile, write_link_cost_profile


def _payload() -> dict[str, object]:
    return {
        "world_size": 4,
        "ranks_per_node": 2,
        "rank_to_node": [0, 0, 1, 1],
        "row_bytes": 4096,
        "edge_slope_us_per_row": [[1.0, 2.0, 8.0, 8.0] for _ in range(4)],
        "edge_intercept_us": [[0.0, 0.5, 4.0, 4.0] for _ in range(4)],
        "wave_launch_us": 0.25,
        "source": "fixture",
        "metadata": {"fixture": True},
    }


def test_link_cost_profile_round_trip_and_planner_config(tmp_path) -> None:
    path = tmp_path / "profile.json"
    written = write_link_cost_profile(path, _payload())
    loaded = load_link_cost_profile(path)
    assert loaded.profile_id == written.profile_id
    assert loaded.planner_config()["ranks_per_node"] == 2
    assert loaded.planner_config()["cost_profile_id"] == written.profile_id


def test_link_cost_profile_rejects_tampering(tmp_path) -> None:
    path = tmp_path / "profile.json"
    write_link_cost_profile(path, _payload())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["edge_slope_us_per_row"][0][1] = 99.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="profile_id"):
        load_link_cost_profile(path)


def test_affine_fit_uses_row_units() -> None:
    slope, intercept = fit_affine_row_cost(
        [(4096, 12.0), (4096 * 4, 18.0), (4096 * 16, 42.0)],
        row_bytes=4096,
    )
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(10.0)


def test_model_row_contract_and_precision(tmp_path):
    from rs.topology import infer_model_row_contract, precision_bytes

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"hidden_size": 128}', encoding="utf-8")
    contract = infer_model_row_contract(model, precision="bf16")
    assert contract["hidden_size"] == 128
    assert contract["element_bytes"] == 2
    assert contract["row_bytes"] == 256
    assert precision_bytes("fp32") == 4


def test_measured_pairwise_profile_does_not_double_count_wave_launch(tmp_path):
    from rs.topology import write_link_cost_profile

    profile = write_link_cost_profile(
        tmp_path / "profile.json",
        {
            "world_size": 2,
            "ranks_per_node": 2,
            "rank_to_node": [0, 0],
            "row_bytes": 256,
            "edge_slope_us_per_row": [[1.0, 2.0], [2.0, 1.0]],
            "edge_intercept_us": [[0.0, 3.0], [3.0, 0.0]],
            "wave_launch_us": 0.0,
        },
    )
    assert profile.wave_launch_us == 0.0
