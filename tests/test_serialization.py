from pathlib import Path

from routesense_poc1.schemas import RunConfig
from routesense_poc1.serialization import load_json, save_json


def test_json_serialization(tmp_path: Path):
    config = RunConfig(model_id="demo", rank=3)
    path = tmp_path / "config.json"
    save_json(config.to_dict(), path)
    data = load_json(path)
    assert data["model_id"] == "demo"
    assert data["rank"] == 3
