from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from rs.core.contracts import PredictionIdentity, TrafficHistoryContext
from rs.core.hashing import stable_hash_dict
from rs.prediction import PredictionEvaluator, PredictionRegistry, PredictionTruth, TrafficPredictionTrainingSample


@dataclass(frozen=True)
class PredictionRolloutSpec:
    train_count: int
    validation_count: int
    test_count: int
    group_key: str = "global"
    history_window: int | None = None
    cold_start_mode: str = "copy_current"


@dataclass(frozen=True)
class PredictionRolloutRecord:
    sample_index: int
    split: str
    predictor_id: str
    fit_sample_count: int
    history_window: int | None
    cold_start_used: bool
    training_digest: str
    predictor_artifact_digest: str
    metrics: dict[str, float]


def run_prediction_rollout(
    *,
    samples: Sequence[TrafficPredictionTrainingSample],
    predictor_id: str,
    rollout_spec: PredictionRolloutSpec,
    predictor_config: dict | None = None,
) -> tuple[PredictionRolloutRecord, ...]:
    sample_list = list(samples)
    for sample in sample_list:
        sample.validate()
    total = int(rollout_spec.train_count + rollout_spec.validation_count + rollout_spec.test_count)
    if total > len(sample_list):
        raise ValueError("rollout split exceeds sample count")
    records: list[PredictionRolloutRecord] = []
    evaluator = PredictionEvaluator()
    train_end = int(rollout_spec.train_count)
    val_end = train_end + int(rollout_spec.validation_count)
    for index, sample in enumerate(sample_list[:total]):
        split = "train" if index < train_end else "validation" if index < val_end else "test"
        prior_train = sample_list[: min(index, train_end)]
        if rollout_spec.history_window is not None:
            history_scope = prior_train[-int(rollout_spec.history_window) :]
        else:
            history_scope = prior_train
        cold_start_used = len(history_scope) == 0
        predictor = PredictionRegistry.create(predictor_id, predictor_config, usage="offline")
        if hasattr(predictor, "fit") and not cold_start_used:
            predictor.fit(history_scope)
        context = TrafficHistoryContext(
            identity=PredictionIdentity(
                request_id=f"rollout:{index}",
                source_layer_id=sample.layer_id,
                target_layer_id=sample.next_layer_id,
            ),
            current_dispatch_rows=sample.current_dispatch_rows,
            current_return_rows=sample.current_return_rows,
            history_dispatch_rows=tuple(item.current_dispatch_rows for item in history_scope),
            world_size=len(sample.current_dispatch_rows),
        )
        if cold_start_used and str(rollout_spec.cold_start_mode) == "copy_current":
            result = PredictionRegistry.create("copy_current", usage="offline").predict(context)
        else:
            result = predictor.predict(context)
        evaluation = evaluator.evaluate(
            result,
            PredictionTruth(actual_dispatch_rows=sample.target_next_dispatch_rows),
        )
        training_digest = stable_hash_dict(
            {
                "training_version": "offline_rollout_v1",
                "predictor_id": predictor_id,
                "group_key": rollout_spec.group_key,
                "history_window": rollout_spec.history_window,
                "training_indices": list(range(max(0, index - len(history_scope)), index)),
            }
        )
        records.append(
            PredictionRolloutRecord(
                sample_index=index,
                split=split,
                predictor_id=str(result.hint.predictor_id),
                fit_sample_count=len(history_scope),
                history_window=rollout_spec.history_window,
                cold_start_used=cold_start_used,
                training_digest=training_digest,
                predictor_artifact_digest=stable_hash_dict(result.to_dict()),
                metrics=dict(evaluation.metrics),
            )
        )
    return tuple(records)


__all__ = ["PredictionRolloutRecord", "PredictionRolloutSpec", "run_prediction_rollout"]
