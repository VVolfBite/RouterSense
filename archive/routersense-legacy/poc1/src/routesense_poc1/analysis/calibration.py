from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from ..core.features import FEATURE_EXTRACTORS
from ..core.schemas import AblationRecord

ROUTING_ONLY_SUBSETS: list[tuple[str, ...]] = [
    ("effective_gate_weight",),
    ("router_probability",),
    ("router_logit",),
    ("abs_router_logit",),
    ("topk_rank",),
    ("top1_top2_gap",),
    ("routing_entropy",),
    ("effective_gate_weight", "topk_rank"),
    ("effective_gate_weight", "top1_top2_gap"),
    ("effective_gate_weight", "routing_entropy"),
    ("router_probability", "topk_rank"),
    ("router_probability", "top1_top2_gap"),
    ("router_probability", "routing_entropy"),
    ("effective_gate_weight", "top1_top2_gap", "routing_entropy"),
    ("router_probability", "top1_top2_gap", "routing_entropy"),
    ("effective_gate_weight", "router_logit", "topk_rank"),
    ("effective_gate_weight", "abs_router_logit", "topk_rank"),
]

STATE_LIKE_SUBSETS: list[tuple[str, ...]] = [
    ("topk_rank",),
    ("layer_id",),
    ("expert_id",),
    ("topk_rank", "layer_id"),
    ("topk_rank", "expert_id"),
    ("layer_id", "expert_id"),
    ("effective_gate_weight", "topk_rank", "layer_id"),
    ("effective_gate_weight", "topk_rank", "expert_id"),
]

COMBINED_SUBSETS: list[tuple[str, ...]] = [
    ("effective_gate_weight", "topk_rank", "layer_id"),
    ("effective_gate_weight", "topk_rank", "expert_id"),
    ("router_probability", "topk_rank", "layer_id"),
    ("router_probability", "topk_rank", "expert_id"),
    ("effective_gate_weight", "top1_top2_gap", "topk_rank", "layer_id"),
    ("router_probability", "top1_top2_gap", "topk_rank", "expert_id"),
    ("effective_gate_weight", "routing_entropy", "topk_rank", "layer_id"),
    ("abs_router_logit", "topk_rank", "expert_id"),
]

FEATURE_FAMILY_SUBSETS: dict[str, list[tuple[str, ...]]] = {
    "routing_only": ROUTING_ONLY_SUBSETS,
    "state_like": STATE_LIKE_SUBSETS,
    "combined": COMBINED_SUBSETS,
}


@dataclass
class CalibratorBundle:
    model: HistGradientBoostingRegressor
    feature_names: list[str]
    feature_family: str
    train_document_ids: list[int]
    validation_document_ids: list[int]
    test_document_ids: list[int]
    selection_metric: str
    selection_score: float


def _feature_matrix(records: list[AblationRecord], feature_names: list[str]) -> list[list[float]]:
    return [[FEATURE_EXTRACTORS[name](record) for name in feature_names] for record in records]


def _split_document_ids(records: list[AblationRecord]) -> tuple[list[int], list[int], list[int]]:
    document_ids = sorted({record.document_id for record in records})
    if len(document_ids) < 3:
        raise ValueError("need at least 3 distinct document_ids for calibration split")
    n = len(document_ids)
    train_end = max(1, int(round(n * 0.7)))
    valid_end = max(train_end + 1, int(round(n * 0.85)))
    if valid_end >= n:
        valid_end = n - 1
    train_ids = document_ids[:train_end]
    valid_ids = document_ids[train_end:valid_end]
    test_ids = document_ids[valid_end:]
    if not valid_ids:
        valid_ids = train_ids[-1:]
        train_ids = train_ids[:-1]
    if not test_ids:
        test_ids = valid_ids[-1:]
        valid_ids = valid_ids[:-1]
    return train_ids, valid_ids, test_ids


def _filter_records(records: list[AblationRecord], document_ids: list[int]) -> list[AblationRecord]:
    allowed = set(document_ids)
    return [record for record in records if record.document_id in allowed]


def _fit_model(
    train_records: list[AblationRecord],
    feature_names: list[str],
    seed: int,
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(random_state=seed)
    model.fit(
        _feature_matrix(train_records, feature_names),
        [record.delta_nll for record in train_records],
    )
    return model


def evaluate_calibrator_bundle(bundle: CalibratorBundle, records: list[AblationRecord]) -> dict:
    features = _feature_matrix(records, bundle.feature_names)
    targets = [record.delta_nll for record in records]
    predictions = bundle.model.predict(features)
    return {
        "mae": float(mean_absolute_error(targets, predictions)),
        "r2": float(r2_score(targets, predictions)),
        "num_records": len(records),
        "feature_names": bundle.feature_names,
        "feature_family": bundle.feature_family,
    }


def _evaluate_feature_subset(
    records: list[AblationRecord],
    feature_names: list[str],
    feature_family: str,
    seed: int,
    train_ids: list[int],
    valid_ids: list[int],
    test_ids: list[int],
) -> dict:
    train_records = _filter_records(records, train_ids)
    valid_records = _filter_records(records, valid_ids)
    test_records = _filter_records(records, test_ids)
    model = _fit_model(train_records, feature_names, seed)
    validation_metrics = evaluate_calibrator_bundle(
        CalibratorBundle(
            model=model,
            feature_names=list(feature_names),
            feature_family="candidate",
            train_document_ids=train_ids,
            validation_document_ids=valid_ids,
            test_document_ids=test_ids,
            selection_metric="validation_mae",
            selection_score=0.0,
        ),
        valid_records,
    )
    test_metrics = evaluate_calibrator_bundle(
        CalibratorBundle(
            model=model,
            feature_names=list(feature_names),
            feature_family="candidate",
            train_document_ids=train_ids,
            validation_document_ids=valid_ids,
            test_document_ids=test_ids,
            selection_metric="validation_mae",
            selection_score=0.0,
        ),
        test_records,
    )
    return {
        "feature_names": list(feature_names),
        "feature_family": feature_family,
        "validation_mae": validation_metrics["mae"],
        "validation_r2": validation_metrics["r2"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "train_records": len(train_records),
        "validation_records": len(valid_records),
        "test_records": len(test_records),
    }


def train_calibrator(records: list[AblationRecord], config, output_dir: Path) -> CalibratorBundle:
    train_ids, valid_ids, test_ids = _split_document_ids(records)
    candidate_results = []
    for feature_family, subsets in FEATURE_FAMILY_SUBSETS.items():
        for feature_names in list(dict.fromkeys(subsets)):
            candidate_results.append(
                _evaluate_feature_subset(
                    records=records,
                    feature_names=list(feature_names),
                    feature_family=feature_family,
                    seed=config.seed,
                    train_ids=train_ids,
                    valid_ids=valid_ids,
                    test_ids=test_ids,
                )
            )
    family_best: dict[str, dict] = {}
    for row in sorted(candidate_results, key=lambda row: (row["validation_mae"], -row["validation_r2"], len(row["feature_names"]))):
        family_best.setdefault(row["feature_family"], row)
    candidate_results.sort(key=lambda row: (row["validation_mae"], -row["validation_r2"], len(row["feature_names"])))
    best = candidate_results[0]
    full_train_records = _filter_records(records, train_ids + valid_ids)
    model = _fit_model(full_train_records, best["feature_names"], config.seed)
    bundle = CalibratorBundle(
        model=model,
        feature_names=list(best["feature_names"]),
        feature_family=str(best["feature_family"]),
        train_document_ids=train_ids,
        validation_document_ids=valid_ids,
        test_document_ids=test_ids,
        selection_metric="validation_mae",
        selection_score=float(best["validation_mae"]),
    )
    joblib.dump(bundle, output_dir / "calibrator.joblib")
    joblib.dump(candidate_results, output_dir / "calibration_candidates.joblib")
    joblib.dump(family_best, output_dir / "calibration_family_best.joblib")
    return bundle


def evaluate_calibrator(bundle: CalibratorBundle, records: list[AblationRecord], output_dir: Path | None = None) -> dict:
    train_records = _filter_records(records, bundle.train_document_ids)
    validation_records = _filter_records(records, bundle.validation_document_ids)
    test_records = _filter_records(records, bundle.test_document_ids)
    metrics = {
        "selected_feature_names": bundle.feature_names,
        "selected_feature_family": bundle.feature_family,
        "selection_metric": bundle.selection_metric,
        "selection_score": bundle.selection_score,
        "train_metrics": evaluate_calibrator_bundle(bundle, train_records),
        "validation_metrics": evaluate_calibrator_bundle(bundle, validation_records),
        "test_metrics": evaluate_calibrator_bundle(bundle, test_records),
        "train_document_ids": bundle.train_document_ids,
        "validation_document_ids": bundle.validation_document_ids,
        "test_document_ids": bundle.test_document_ids,
    }
    if output_dir is not None:
        candidate_results = joblib.load(output_dir / "calibration_candidates.joblib")
        metrics["candidate_feature_sets"] = candidate_results
        metrics["family_best_feature_sets"] = joblib.load(output_dir / "calibration_family_best.joblib")
    return metrics
