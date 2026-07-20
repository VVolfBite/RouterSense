#!/usr/bin/env python3
"""Train a lightweight offline traffic predictor from replay fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.runtime.offline.prediction import (
    FATEStyleHistoryPredictor,
    FATEStyleLinearTrafficPredictor,
    PredictorArtifact,
    save_predictor_artifact,
)
from rs.runtime.offline.prediction.feature_builder import load_fixture_samples


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--output-artifact", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--predictor", choices=("linear", "history"), required=True)
    parser.add_argument("--train-split", choices=("all", "first_half_second_half"), default="all")
    return parser.parse_args()


def _build_predictor(name: str):
    if name == "linear":
        return FATEStyleLinearTrafficPredictor()
    return FATEStyleHistoryPredictor()


def main() -> None:
    args = _parse_args()
    samples = load_fixture_samples(Path(args.fixture_dir))
    if args.train_split == "first_half_second_half" and len(samples) > 1:
        train_samples = samples[: max(1, len(samples) // 2)]
    else:
        train_samples = samples
    predictor = _build_predictor(str(args.predictor)).fit(train_samples)
    artifact = predictor.to_artifact()
    save_predictor_artifact(args.output_artifact, artifact)
    summary = {
        "predictor_name": artifact.predictor_name,
        "predictor_version": artifact.predictor_version,
        "feature_spec": artifact.feature_spec,
        "world_size": artifact.world_size,
        "train_split": str(args.train_split),
        "train_sample_count": len(train_samples),
        "fixture_sample_count": len(samples),
    }
    Path(args.output_summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
