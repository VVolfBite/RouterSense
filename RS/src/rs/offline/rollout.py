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

    def validate(self) -> None:
        if int(self.train_count) < 0 or int(self.validation_count) < 0 or int(self.test_count) < 0:
            raise ValueError("rollout counts must be >= 0")
        if self.history_window is not None and int(self.history_window) <= 0:
            raise ValueError("history_window must be > 0 when provided")
        if str(self.cold_start_mode) not in {"copy_current", "predictor"}:
            raise ValueError("unsupported cold_start_mode")
        if not str(self.group_key).strip():
            raise ValueError("group_key must be non-empty")


@dataclass(frozen=True)
class PredictionRolloutSample:
    sample_id: str
    group_id: str
    sequence_index: int
    training_sample: TrafficPredictionTrainingSample

    def validate(self) -> None:
        if not str(self.sample_id).strip():
            raise ValueError("sample_id must be non-empty")
        if not str(self.group_id).strip():
            raise ValueError("group_id must be non-empty")
        if int(self.sequence_index) < 0:
            raise ValueError("sequence_index must be >= 0")
        self.training_sample.validate()


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


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _json_safe(to_dict())
        except Exception:
            return repr(value)
    return repr(value)


def _normalize_samples(
    samples: Sequence[TrafficPredictionTrainingSample | PredictionRolloutSample],
    *,
    group_key: str,
) -> list[PredictionRolloutSample]:
    normalized: list[PredictionRolloutSample] = []
    for index, item in enumerate(samples):
        if isinstance(item, PredictionRolloutSample):
            item.validate()
            normalized.append(item)
            continue
        item.validate()
        normalized.append(
            PredictionRolloutSample(
                sample_id=f"sample:{index}",
                group_id=str(group_key),
                sequence_index=index,
                training_sample=item,
            )
        )
    return normalized


def run_prediction_rollout(
    *,
    samples: Sequence[TrafficPredictionTrainingSample | PredictionRolloutSample],
    predictor_id: str,
    rollout_spec: PredictionRolloutSpec,
    predictor_config: dict | None = None,
) -> tuple[PredictionRolloutRecord, ...]:
    rollout_spec.validate()
    sample_list = _normalize_samples(samples, group_key=rollout_spec.group_key)
    total = int(rollout_spec.train_count + rollout_spec.validation_count + rollout_spec.test_count)
    if total > len(sample_list):
        raise ValueError("rollout split exceeds sample count")
    records: list[PredictionRolloutRecord] = []
    evaluator = PredictionEvaluator()
    train_end = int(rollout_spec.train_count)
    val_end = train_end + int(rollout_spec.validation_count)
    for index, sample in enumerate(sample_list[:total]):
        split = "train" if index < train_end else "validation" if index < val_end else "test"
        prior_train = [
            candidate
            for prior_index, candidate in enumerate(sample_list[:train_end])
            if prior_index < index
            and candidate.group_id == sample.group_id
            and int(candidate.sequence_index) < int(sample.sequence_index)
        ]
        if rollout_spec.history_window is not None:
            history_scope = prior_train[-int(rollout_spec.history_window) :]
        else:
            history_scope = prior_train
        cold_start_used = len(history_scope) == 0
        predictor = PredictionRegistry.create(predictor_id, predictor_config, usage="offline")
        if hasattr(predictor, "fit") and not cold_start_used:
            predictor.fit(tuple(item.training_sample for item in history_scope))
        current = sample.training_sample
        context = TrafficHistoryContext(
            identity=PredictionIdentity(
                request_id=f"rollout:{index}",
                source_layer_id=current.layer_id,
                target_layer_id=current.next_layer_id,
            ),
            current_dispatch_rows=current.current_dispatch_rows,
            current_return_rows=current.current_return_rows,
            history_dispatch_rows=tuple(item.training_sample.current_dispatch_rows for item in history_scope),
            world_size=len(current.current_dispatch_rows),
        )
        if cold_start_used and str(rollout_spec.cold_start_mode) == "copy_current":
            result = PredictionRegistry.create("copy_current", usage="offline").predict(context)
        else:
            result = predictor.predict(context)
        evaluation = evaluator.evaluate(
            result,
            PredictionTruth(actual_dispatch_rows=current.target_next_dispatch_rows),
        )
        training_ids = tuple(item.sample_id for item in history_scope)
        training_digest = stable_hash_dict(
            {
                "training_version": "offline_rollout_v1",
                "predictor_id": predictor_id,
                "group_key": rollout_spec.group_key,
                "group_id": sample.group_id,
                "history_window": rollout_spec.history_window,
                "training_sample_ids": list(training_ids),
            }
        )
        predictor_artifact_digest = stable_hash_dict(
            {
                "predictor_id": predictor_id,
                "predictor_config": dict(predictor_config or {}),
                "training_sample_ids": list(training_ids),
                "training_data": [
                    {
                        "sample_id": item.sample_id,
                        "dispatch": [list(row) for row in item.training_sample.current_dispatch_rows],
                        "target": [list(row) for row in item.training_sample.target_next_dispatch_rows],
                    }
                    for item in history_scope
                ],
                "fitted_artifact": _json_safe(dict(getattr(predictor, "__dict__", {}))),
                "result_hint": result.hint.to_dict(),
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
                predictor_artifact_digest=predictor_artifact_digest,
                metrics=dict(evaluation.metrics),
            )
        )
    return tuple(records)


__all__ = ["PredictionRolloutRecord", "PredictionRolloutSample", "PredictionRolloutSpec", "run_prediction_rollout"]
