from __future__ import annotations

import json
from pathlib import Path

from rs_sim.trace.collection.config import example_config, validate_pipeline_config
from rs_sim.trace.collection.fixture_builder import build_fixtures_from_capture
from rs_sim.trace.collection.pipeline import simulate_fixture
from rs_sim.trace.schema.fixtures import build_builtin_fixtures


def test_captured_counts_build_fixture_and_run_formal_simulation(tmp_path: Path):
    source = build_builtin_fixtures()[0]
    config = example_config()
    config["output_dir"] = str(tmp_path)
    config["capture"]["capture_id"] = "pipeline-smoke"
    config["capture"]["request_id"] = source.windows[0].request_id
    config["capture"]["sample_id_prefix"] = "pipeline-sample"
    config["capture"]["model_path"] = str(tmp_path / "model")
    config["capture"]["rank_to_node"] = list(source.windows[0].mapping.rank_to_node)
    config["capture"]["expert_to_rank"] = list(source.windows[0].mapping.expert_to_rank)
    for phase, attr in (("dispatch", "dispatch_payload_spec"), ("combine", "combine_payload_spec")):
        spec = getattr(source.windows[0], attr)
        config["payload"][phase].update(
            {
                "token_payload_bytes_per_row": spec.token_payload_bytes_per_row,
                "auxiliary_payload_bytes_per_row": spec.auxiliary_payload_bytes_per_row,
                "metadata_bytes_per_edge": spec.metadata_bytes_per_edge,
                "alignment_bytes": spec.alignment_bytes,
                "padding_rule": spec.padding_rule,
                "dtype": spec.dtype,
            }
        )
    descriptor = source.windows[0].descriptor_metadata_spec
    config["payload"]["descriptor"] = {
        "fixed_header_bytes": descriptor.fixed_header_bytes,
        "per_destination_entry_bytes": descriptor.per_destination_entry_bytes,
    }
    config["fixture"]["compute_fallback_ns"] = {
        "combine_release_to_router_ready_ns": 1,
        "router_and_pack_ns": 1,
        "dispatch_local_postprocess_ns": 1,
        "dispatch_release_to_combine_source_ready_ns": 1,
        "bootstrap_router_and_pack_ns": 1,
    }
    config["simulation"].update(
        {
            "algorithm": "local(event(fifo()))",
            "information": "ZERO_P2",
            "overlap": "OVERLAP",
            "release": "PHASE_BARRIER",
        }
    )
    config = validate_pipeline_config(config)
    raw = tmp_path / "raw"
    raw.mkdir()
    for rank in range(source.world_size):
        rows = []
        for window in source.windows:
            rows.append(
                {
                    "sample_id": "pipeline-sample:step0",
                    "request_id": window.request_id,
                    "decode_step": 0,
                    "layer_id": window.layer_id,
                    "world_size": window.mapping.world_size,
                    "num_experts": window.mapping.num_experts,
                    "source_rank": rank,
                    "raw_selected_rows": list(window.routing.raw_selected_rows[rank]),
                    "kept_rows": list(window.routing.kept_rows[rank]),
                    "dropped_rows": list(window.routing.dropped_rows[rank]),
                    "padding_rows": list(window.routing.padding_rows[rank]),
                    "expert_to_rank_map": list(window.mapping.expert_to_rank),
                    "rank_to_node": list(window.mapping.rank_to_node),
                }
            )
        path = raw / f"rank{rank:04d}-global{rank:04d}_source_expert_counts.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    fixture_paths = build_fixtures_from_capture(config)
    result_path = simulate_fixture(config, fixture_paths[0])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "PASS"
    assert result["transport_transport"] is True
    assert result["axes"]["planning_window"] == "P12"
