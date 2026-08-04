from __future__ import annotations

import json
from pathlib import Path

import pytest

from rs_sim.trace.collection.config import example_config, validate_pipeline_config
from rs_sim.trace.collection.extract import extract_routing_counts
from rs_sim.trace.collection.fixture_builder import build_fixtures_from_capture
from rs_sim.trace.io.serialization import load_fixture
from rs_sim.trace.schema.validation import validate_fixture


class FakeTensor:
    def __init__(self, value):
        self.value = value
        self.shape = _shape(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


def _shape(value):
    result = []
    cursor = value
    while isinstance(cursor, list):
        result.append(len(cursor))
        cursor = cursor[0] if cursor else None
    return tuple(result)


def test_extract_final_map_and_explicit_padding():
    routing = FakeTensor(
        [
            [True, False, False, True],
            [True, True, False, False],
            [False, True, True, False],
        ]
    )
    result = extract_routing_counts(
        routing_map=routing,
        explicit_padding_rows=(1, 0, 0, 0),
        drop_and_pad=True,
    )
    assert result.kept_rows == (1, 2, 1, 1)
    assert result.padding_rows == (1, 0, 0, 0)
    assert result.raw_selected_rows == result.kept_rows
    assert result.dropped_rows == (0, 0, 0, 0)


def test_extract_3d_rank_instance_map_flattens_deterministically():
    routing = FakeTensor(
        [
            [[True, False], [False, True]],
            [[False, True], [True, False]],
        ]
    )
    result = extract_routing_counts(routing_map=routing)
    assert result.source_shape == (2, 2, 2)
    assert result.kept_rows == (1, 1, 1, 1)


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_capture_artifacts_finalize_into_valid_fixture(tmp_path: Path):
    config = example_config()
    config["output_dir"] = str(tmp_path)
    config["capture"]["capture_id"] = "capture-test"
    config["capture"]["request_id"] = "request-test"
    config["capture"]["sample_id_prefix"] = "sample-test"
    config["capture"]["model_path"] = str(tmp_path / "model")
    config["capture"]["expert_to_rank"] = [0, 1, 2, 3]
    config["payload"]["dispatch"]["padding_rule"] = "EDGE_TOTAL_ALIGN_UP"
    config["payload"]["combine"]["padding_rule"] = "EDGE_TOTAL_ALIGN_UP"
    config = validate_pipeline_config(config)

    raw = tmp_path / "raw"
    for rank in range(4):
        routing_rows = []
        compute_rows = []
        for layer in (0, 1):
            counts = [0, 0, 0, 0]
            counts[(rank + layer + 1) % 4] = 2 + rank
            routing_rows.append(
                {
                    "sample_id": "sample-test:step0",
                    "request_id": "request-test",
                    "decode_step": 0,
                    "layer_id": layer,
                    "world_size": 4,
                    "num_experts": 4,
                    "source_rank": rank,
                    "raw_selected_rows": counts,
                    "kept_rows": counts,
                    "dropped_rows": [0, 0, 0, 0],
                    "padding_rows": [0, 0, 0, 0],
                    "expert_to_rank_map": [0, 1, 2, 3],
                    "rank_to_node": [0, 0, 0, 0],
                }
            )
            compute_rows.append(
                {
                    "request_id": "request-test",
                    "decode_step": 0,
                    "layer_id": layer,
                    "source_rank": rank,
                    "field_values_ns": {
                        "combine_release_to_router_ready_ns": 100 + rank,
                        "router_and_pack_ns": 20 + rank,
                        "dispatch_local_postprocess_ns": 10 + rank,
                        "dispatch_release_to_combine_source_ready_ns": 200 + rank,
                        "bootstrap_router_and_pack_ns": 30 + rank,
                    },
                }
            )
        _write_jsonl(raw / f"rank{rank:04d}-global{rank:04d}_source_expert_counts.jsonl", routing_rows)
        _write_jsonl(raw / f"rank{rank:04d}-global{rank:04d}_local_compute.jsonl", compute_rows)

    paths = build_fixtures_from_capture(config)
    assert len(paths) == 1
    fixture = load_fixture(paths[0])
    report = validate_fixture(fixture)
    assert report["status"] == "PASS"
    assert len(fixture.windows) == 2
    assert fixture.windows[0].is_bootstrap_p0 is True
    assert fixture.windows[1].is_bootstrap_p0 is False
    assert fixture.windows[0].local_compute.router_and_pack_ns == (20, 21, 22, 23)


def test_megatron_auto_adapter_with_fake_lifecycle(tmp_path: Path, monkeypatch):
    import sys
    import types
    import rs_sim.trace.collection.api as capture_api
    import rs_sim.trace.collection.megatron as capture_megatron

    config = example_config()
    config["output_dir"] = str(tmp_path / "output")
    config["capture"]["capture_id"] = "fake-megatron"
    config["capture"]["request_id"] = "fake-request"
    config["capture"]["sample_id_prefix"] = "fake-sample"
    config["capture"]["model_path"] = str(tmp_path / "model")
    config["capture"]["rank_to_node"] = [0]
    config["capture"]["expert_to_rank"] = [0, 0]
    config_path = tmp_path / "capture.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("RS_SIM_CAPTURE_CONFIG", str(config_path))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    capture_api._SESSION = None
    capture_megatron._PATCHED = False

    package_names = [
        "megatron",
        "megatron.core",
        "megatron.core.transformer",
        "megatron.core.transformer.moe",
    ]
    for name in package_names:
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    dispatcher_module = types.ModuleType("megatron.core.transformer.moe.token_dispatcher")
    layer_module = types.ModuleType("megatron.core.transformer.moe.moe_layer")

    class FakeDispatcher:
        __module__ = dispatcher_module.__name__
        local_expert_indices = [0, 1]
        drop_and_pad = False

        def dispatch_postprocess(self, hidden_states, probs):
            return hidden_states, (1, 1), probs

    class FakeMoELayer:
        __module__ = layer_module.__name__

        def __init__(self):
            self.layer_number = 0
            self.token_dispatcher = FakeDispatcher()

        def route(self, hidden_states):
            return [[0.7, 0.0], [0.0, 0.8]], [[True, False], [False, True]]

        def preprocess(self, hidden_states, probs, routing_map):
            return hidden_states, probs

        def routed_experts_compute(self, hidden_states, probs):
            self.token_dispatcher.dispatch_postprocess(hidden_states, probs)
            return hidden_states, None

        def forward(self, hidden_states):
            probs, routing = self.route(hidden_states)
            hidden_states, probs = self.preprocess(hidden_states, probs, routing)
            return self.routed_experts_compute(hidden_states, probs)

    dispatcher_module.FakeDispatcher = FakeDispatcher
    layer_module.FakeMoELayer = FakeMoELayer
    monkeypatch.setitem(sys.modules, dispatcher_module.__name__, dispatcher_module)
    monkeypatch.setitem(sys.modules, layer_module.__name__, layer_module)

    result = capture_megatron.install_megatron_auto_capture()
    assert result["status"] == "INSTALLED"
    import torch
    FakeMoELayer().forward(torch.tensor([[1.0], [2.0]], dtype=torch.float32))
    capture_api.flush_capture()

    files = list((tmp_path / "output" / "raw").glob("*_source_expert_counts.jsonl"))
    assert len(files) == 1
    rows = [json.loads(line) for line in files[0].read_text().splitlines() if line]
    assert rows[0]["kept_rows"] == [1, 1]
    assert rows[0]["capture_quality"] == "exact_realized_final_map_raw_drop_unavailable"


def test_capture_pause_excludes_warmup_and_sample_boundary_drops_final_interval(tmp_path: Path, monkeypatch):
    import rs_sim.trace.collection.api as capture_api

    config = example_config()
    config["output_dir"] = str(tmp_path / "output")
    config["capture"]["capture_id"] = "boundary-test"
    config["capture"]["request_id"] = "boundary-test"
    config["capture"]["sample_id_prefix"] = "boundary-test"
    config["capture"]["model_path"] = str(tmp_path / "model")
    config["capture"]["rank_to_node"] = [0]
    config["capture"]["expert_to_rank"] = [0, 0]
    config_path = tmp_path / "capture.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("RS_SIM_CAPTURE_CONFIG", str(config_path))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    capture_api._SESSION = None

    session = capture_api.current_capture_session(required=True)
    assert session is not None
    capture_api.set_capture_enabled(False)
    session.record_routing(layer_id=0, routing_map=[[True, False]], decode_step=-1)

    capture_api.set_capture_enabled(True)
    capture_api.set_capture_context(request_id="sample-0", decode_step=0)
    session.record_routing(layer_id=0, routing_map=[[True, False]], decode_step=0)
    session.mark_moe_output_ready(layer_id=0, decode_step=0)
    capture_api.finish_capture_sample(decode_step=0)

    capture_api.set_capture_context(request_id="sample-1", decode_step=1)
    session.record_routing(layer_id=0, routing_map=[[False, True]], decode_step=1)
    session.mark_moe_output_ready(layer_id=0, decode_step=1)
    capture_api.finish_capture_sample(decode_step=1)
    capture_api.flush_capture()

    routing_file = next((tmp_path / "output" / "raw").glob("*_source_expert_counts.jsonl"))
    routing = [json.loads(line) for line in routing_file.read_text().splitlines() if line]
    assert [row["decode_step"] for row in routing] == [0, 1]
    compute_files = list((tmp_path / "output" / "raw").glob("*_local_compute.jsonl"))
    if compute_files:
        compute = [json.loads(line) for line in compute_files[0].read_text().splitlines() if line]
        assert all("combine_release_to_router_ready_ns" not in row["field_values_ns"] for row in compute)


def test_megatron_core_015_split_lifecycle_records_exact_routing(tmp_path: Path, monkeypatch):
    import sys
    import types
    import rs_sim.trace.collection.api as capture_api
    import rs_sim.trace.collection.megatron as capture_megatron

    config = example_config()
    config["output_dir"] = str(tmp_path / "output015")
    config["capture"]["capture_id"] = "fake-megatron-015"
    config["capture"]["request_id"] = "fake-request-015"
    config["capture"]["sample_id_prefix"] = "fake-sample-015"
    config["capture"]["model_path"] = str(tmp_path / "model")
    config["capture"]["rank_to_node"] = [0]
    config["capture"]["expert_to_rank"] = [0, 0]
    config_path = tmp_path / "capture015.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("RS_SIM_CAPTURE_CONFIG", str(config_path))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    capture_api._SESSION = None
    capture_megatron._PATCHED = False

    for name in [
        "megatron",
        "megatron.core",
        "megatron.core.transformer",
        "megatron.core.transformer.moe",
    ]:
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    dispatcher_module = types.ModuleType("megatron.core.transformer.moe.token_dispatcher")
    layer_module = types.ModuleType("megatron.core.transformer.moe.moe_layer")

    class FakeDispatcher:
        __module__ = dispatcher_module.__name__
        local_expert_indices = [0, 1]
        drop_and_pad = False

        def dispatch_preprocess(self, tokens, routing_map, probs):
            self.routing_map = routing_map
            return tokens, probs

        def dispatch_postprocess(self, hidden_states, probs):
            return hidden_states, (1, 1), probs

    class FakeMoELayer:
        __module__ = layer_module.__name__

        def __init__(self):
            self.layer_number = 0
            self.token_dispatcher = FakeDispatcher()

        def router_and_preprocess(self, hidden_states):
            probs = FakeTensor([[0.7, 0.0], [0.0, 0.8]])
            routing = FakeTensor([[True, False], [False, True]])
            hidden_states, probs = self.token_dispatcher.dispatch_preprocess(hidden_states, routing, probs)
            return hidden_states, probs, hidden_states

        def routed_experts_compute(self, hidden_states, probs, residual):
            self.token_dispatcher.dispatch_postprocess(hidden_states, probs)
            return hidden_states, None

        def forward(self, hidden_states):
            hidden_states, probs, residual = self.router_and_preprocess(hidden_states)
            return self.routed_experts_compute(hidden_states, probs, residual)

    dispatcher_module.FakeDispatcher = FakeDispatcher
    layer_module.FakeMoELayer = FakeMoELayer
    monkeypatch.setitem(sys.modules, dispatcher_module.__name__, dispatcher_module)
    monkeypatch.setitem(sys.modules, layer_module.__name__, layer_module)

    result = capture_megatron.install_megatron_auto_capture()
    assert result["status"] == "INSTALLED"
    assert "SPLIT_ROUTER_DISPATCH" in result["lifecycle_profiles"]
    import torch
    FakeMoELayer().forward(torch.tensor([[1.0], [2.0]], dtype=torch.float32))
    capture_api.flush_capture()

    files = list((tmp_path / "output015" / "raw").glob("*_source_expert_counts.jsonl"))
    assert len(files) == 1
    rows = [json.loads(line) for line in files[0].read_text().splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["kept_rows"] == [1, 1]
    assert rows[0]["metadata"]["adapter_stage"] == "DISPATCHER_DISPATCH_PREPROCESS"


def test_completed_invocation_step_preserves_explicit_sample_context(tmp_path: Path, monkeypatch):
    import rs_sim.trace.collection.api as capture_api

    config = example_config()
    config["output_dir"] = str(tmp_path / "output-step")
    config["capture"]["capture_id"] = "explicit-step"
    config["capture"]["request_id"] = "explicit-step"
    config["capture"]["sample_id_prefix"] = "explicit-step"
    config["capture"]["rank_to_node"] = [0]
    config["capture"]["expert_to_rank"] = [0, 0]
    config_path = tmp_path / "capture-step.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("RS_SIM_CAPTURE_CONFIG", str(config_path))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    capture_api._SESSION = None

    session = capture_api.current_capture_session(required=True)
    assert session is not None
    session.set_context(request_id="sample-7", decode_step=7)
    assert session.completed_invocation_step(1) == 7
    session.set_context(request_id="implicit", decode_step=None)
    assert session.invocation_step(1, advance=True) == 0
    assert session.completed_invocation_step(1) == 0
