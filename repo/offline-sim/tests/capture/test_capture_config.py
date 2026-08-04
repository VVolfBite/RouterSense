from __future__ import annotations

from rs_sim.trace.collection.config import example_config, validate_pipeline_config


def test_example_config_validates():
    value = validate_pipeline_config(example_config())
    assert value["capture"]["backend"] == "MEGATRON_CORE_AUTO"
    assert value["payload"]["dispatch"]["padding_rule"] == "EDGE_TOTAL_ALIGN_UP"
    assert value["simulation"]["algorithm"] == "joint(global_(rscf()))"
    assert value["simulation"]["release"] == "RANK_LOCAL"
    assert value["simulation"]["p0_p1_compute_end_barrier"] is True


def test_online_fate_config_requires_megatron_backend_but_no_external_path():
    config = example_config()
    config["prediction"] = {
        "mode": "FATE_P2",
        "provider": "MEGATRON_SAMPLED_FATE",
        "max_sample_tokens": 2048,
    }
    value = validate_pipeline_config(config)
    assert value["prediction"]["mode"] == "FATE_P2"
    assert value["prediction"]["provider"] == "MEGATRON_SAMPLED_FATE"
    assert value["prediction"]["fate_artifact_path"] is None


def test_external_fate_config_requires_artifact_path():
    import pytest
    from rs_sim.trace.collection.config import CaptureConfigError

    config = example_config()
    config["prediction"] = {"mode": "FATE_P2", "provider": "EXTERNAL_ARTIFACT"}
    with pytest.raises(CaptureConfigError, match="fate_artifact_path"):
        validate_pipeline_config(config)
