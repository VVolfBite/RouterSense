from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import json
import yaml

from rs.experiments.specs import ExperimentSpec, PlanningCase, RunKind, SuiteSpec
from rs.scheduling.catalog import resolve_algorithm_id


class UnsupportedLegacyExperimentConfig(ValueError):
    pass


@dataclass(frozen=True)
class LoadedExperimentConfig:
    spec: ExperimentSpec
    resolved_config_path: Path
    resolved_config_yaml: str
    migration_report: dict[str, object]

    @property
    def config_digest(self) -> str:
        return self.spec.config_digest()


class ExperimentConfigLoader:
    schema_version = 2

    def load(self, *, config_path: str | Path) -> LoadedExperimentConfig:
        path = Path(config_path).resolve()
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("experiment config must decode to a mapping")
        migrated, migration_report = self._migrate(payload)
        spec = self._build_spec(migrated)
        spec.validate()
        resolved_yaml = yaml.safe_dump(spec.to_dict(), sort_keys=False, allow_unicode=False)
        return LoadedExperimentConfig(
            spec=spec,
            resolved_config_path=path,
            resolved_config_yaml=resolved_yaml,
            migration_report=migration_report,
        )

    def write_resolved_artifacts(self, loaded: LoadedExperimentConfig, *, output_dir: str | Path) -> None:
        target_dir = Path(output_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "resolved_config.yaml").write_text(loaded.resolved_config_yaml, encoding="utf-8")
        (target_dir / "migration_report.json").write_text(
            json.dumps(loaded.migration_report, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )

    def _migrate(self, payload: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
        schema_version = int(payload.get("schema_version", 1))
        if schema_version == self.schema_version:
            return dict(payload), {
                "source_schema_version": schema_version,
                "target_schema_version": self.schema_version,
                "migrated": False,
                "changes": [],
            }
        if schema_version != 1:
            raise ValueError(f"unsupported experiment schema_version {schema_version!r}")
        raise UnsupportedLegacyExperimentConfig(
            "schema v1 experiment config is unsupported without a field-preserving migration; rewrite to schema_version: 2"
        )

    def _build_spec(self, payload: Mapping[str, object]) -> ExperimentSpec:
        unknown = set(payload) - {"schema_version", "experiment_id", "suites", "planning_cases", "defaults"}
        if unknown:
            raise ValueError(f"unknown top-level experiment config keys: {sorted(unknown)}")
        suites_raw = payload.get("suites", ())
        cases_raw = payload.get("planning_cases", ())
        if not isinstance(suites_raw, list):
            raise ValueError("suites must be a list")
        if not isinstance(cases_raw, list):
            raise ValueError("planning_cases must be a list")
        suites = tuple(self._build_suite(item) for item in suites_raw)
        cases = tuple(self._build_case(item) for item in cases_raw)
        defaults = payload.get("defaults", {})
        if defaults is not None and not isinstance(defaults, Mapping):
            raise ValueError("defaults must be a mapping")
        return ExperimentSpec(
            schema_version=int(payload.get("schema_version", 0)),
            experiment_id=str(payload.get("experiment_id", "")),
            suites=suites,
            planning_cases=cases,
            defaults=dict(defaults or {}),
        )

    def _build_suite(self, payload: object) -> SuiteSpec:
        if not isinstance(payload, Mapping):
            raise ValueError("suite entry must be a mapping")
        unknown = set(payload) - {"suite_id", "markers", "case_ids", "run_kinds", "description"}
        if unknown:
            raise ValueError(f"unknown suite keys: {sorted(unknown)}")
        return SuiteSpec(
            suite_id=str(payload.get("suite_id", "")),
            markers=tuple(str(item) for item in payload.get("markers", ())),
            case_ids=tuple(str(item) for item in payload.get("case_ids", ())),
            run_kinds=tuple(RunKind(str(item)) for item in payload.get("run_kinds", ())),
            description=str(payload.get("description", "")),
        )

    def _build_case(self, payload: object) -> PlanningCase:
        if not isinstance(payload, Mapping):
            raise ValueError("planning case entry must be a mapping")
        unknown = set(payload) - {
            "case_id",
            "run_kind",
            "planner_id",
            "planner_family",
            "selector_mode",
            "predictor_id",
            "prediction_mode",
            "execution_backend",
            "instrumentation_mode",
            "fallback_policy",
        }
        if unknown:
            raise ValueError(f"unknown planning case keys: {sorted(unknown)}")
        case = PlanningCase(
            case_id=str(payload.get("case_id", "")),
            run_kind=RunKind(str(payload.get("run_kind", ""))),
            planner_id=str(payload.get("planner_id", "")),
            planner_family=str(payload.get("planner_family", "")),
            selector_mode=str(payload.get("selector_mode", "")),
            predictor_id=str(payload.get("predictor_id", "")),
            prediction_mode=str(payload.get("prediction_mode", "")),
            execution_backend=str(payload.get("execution_backend", "")),
            instrumentation_mode=str(payload.get("instrumentation_mode", "")),
            fallback_policy=str(payload.get("fallback_policy", "")),
        )
        if case.run_kind == RunKind.OFFLINE_EVALUATION:
            resolve_algorithm_id(case.planner_id)
        return case
