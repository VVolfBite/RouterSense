from routesense_poc1.schemas import (
    AblationResult,
    CorrectnessCheckResult,
    EnvironmentInfo,
    MoELayerInfo,
    RouterTrace,
    RouteSnapshot,
    RunConfig,
)


def test_schema_json_round_trip():
    config = RunConfig(model_id="demo", rank=1, output_dir="tmp")
    payload = config.to_dict()
    restored = RunConfig.from_dict(payload)
    assert restored.model_id == config.model_id
    assert restored.rank == config.rank

    trace = RouterTrace(layer_path="layer", layer_id=2, token_pos=3, target_token_id=4, target_token_text="x")
    assert RouterTrace.from_dict(trace.to_dict()).layer_path == trace.layer_path

    layer_info = MoELayerInfo("m", "Foo", "(x)", True, True)
    assert MoELayerInfo.from_dict(layer_info.to_dict()).has_gate is True

    snapshot = RouteSnapshot("l", 0, [1, 2], [0.7, 0.3])
    assert RouteSnapshot.from_dict(snapshot.to_dict()).weights == [0.7, 0.3]

    result = AblationResult("m", "l", 0, 1, 0, 2, 0.1, 0.2, 0.1, [0.7, 0.3], [0.0, 0.3], True)
    assert AblationResult.from_dict(result.to_dict()).delta_nll == 0.1

    check = CorrectnessCheckResult(True, "ok", {"x": 1})
    assert CorrectnessCheckResult.from_dict(check.to_dict()).details["x"] == 1


def test_environment_info_round_trip():
    info = EnvironmentInfo(cuda_available=False, device="cpu")
    restored = EnvironmentInfo.from_dict(info.to_dict())
    assert restored.device == "cpu"
