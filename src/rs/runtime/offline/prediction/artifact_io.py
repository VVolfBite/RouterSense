from __future__ import annotations

import json
from pathlib import Path

from .contracts import PredictorArtifact


def save_predictor_artifact(path: str | Path, artifact: PredictorArtifact) -> None:
    Path(path).write_text(json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_predictor_artifact(path: str | Path) -> PredictorArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return PredictorArtifact(
        predictor_name=str(payload["predictor_name"]),
        predictor_version=str(payload["predictor_version"]),
        feature_spec=str(payload["feature_spec"]),
        world_size=int(payload["world_size"]),
        metadata=dict(payload.get("metadata", {})),
        payload=dict(payload.get("payload", {})),
    )
