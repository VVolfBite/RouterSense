from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from rs_sim.runtime.config.profiles import load_runtime_profile_bundle_json
from rs_sim.runtime.core.engine import _default_topology_for_fixture, _rscf_wire_cost_model_from_runtime
from rs_sim.scheduler.core.oracle import solve_exact_wire
from rs_sim.scheduler.planning.planner import FairnessContract, SchedulingProblem, SchedulingTask
from rs_sim.scheduler.stable import stable_digest
from rs_sim.trace import load_fixture

CHUNK_BYTES = 262_144


def _tasks_from_matrix(*, fixture_id: str, window_index: int, phase: int, matrix: Iterable[Iterable[int]]) -> tuple[SchedulingTask, ...]:
    tasks: list[SchedulingTask] = []
    for src, row in enumerate(matrix):
        for dst, raw in enumerate(row):
            total = int(raw)
            if src == dst or total <= 0:
                continue
            offset = 0
            chunk = 0
            while offset < total:
                payload = min(CHUNK_BYTES, total - offset)
                task_id = f"{fixture_id}:w{window_index}:p{phase}:{src}->{dst}:c{chunk}:o{offset}:b{payload}"
                tasks.append(SchedulingTask(
                    task_id=task_id,
                    phase_token=f"P{phase}",
                    phase_ordinal=phase,
                    src_rank=src,
                    dst_rank=dst,
                    payload_bytes=payload,
                    chunk_index=chunk,
                    byte_offset=offset,
                    ready_at_ns=0,
                ))
                offset += payload
                chunk += 1
    return tuple(tasks)


def _problem(rank_count: int, tasks: tuple[SchedulingTask, ...], *, tag: str) -> SchedulingProblem:
    fairness = FairnessContract(
        task_catalogue_digest=stable_digest(tuple(task.task_id for task in tasks)),
        task_boundary_digest=stable_digest(tuple(
            (task.task_id, task.phase_token, task.src_rank, task.dst_rank,
             task.chunk_index, task.byte_offset, task.payload_bytes)
            for task in tasks
        )),
        taskization_digest=stable_digest({"chunk_bytes": CHUNK_BYTES}),
        receiver_contract_rule_digest=stable_digest("receiver"),
        buffer_profile_digest=stable_digest("buffer"),
        compiler_digest=stable_digest("compiler"),
        transport_digest=stable_digest("transport"),
        release_model_digest=stable_digest(tag),
        information_digest=stable_digest(tag),
        cost_model_digest=stable_digest("wire"),
    )
    return SchedulingProblem(
        rank_count=rank_count,
        tasks=tasks,
        phase_tokens=tuple(dict.fromkeys(task.phase_token for task in tasks)),
        fairness=fairness,
    )


def _result_row(base: dict[str, Any], scope: str, phase: str, result: Any) -> dict[str, Any]:
    return {
        **base,
        "scope": scope,
        "phase": phase,
        "solver_status": result.solver_status,
        "solver_backend": result.solver_backend,
        "certified_optimal": bool(result.certified_optimal),
        "objective_ns": result.objective_units,
        "best_bound_ns": result.best_bound,
        "gap_pct": None if result.optimality_gap is None else 100.0 * float(result.optimality_gap),
        "solve_time_ms": result.solve_time_ms,
        "canonical_task_count": result.canonical_task_count,
        "symmetry_group_count": result.symmetry_group_count,
        "candidate_slot_count": result.candidate_slot_count,
        "variable_count": result.variable_count,
        "constraint_count": result.constraint_count,
        "search_nodes": result.search_nodes,
        "wave_count": len(result.waves),
        "incumbent_source": result.incumbent_source,
        "failure_reason": result.failure_reason,
    }


def _coordinates(path: Path) -> tuple[str, int, int, int]:
    parts = path.parts
    model = "unknown"
    ep = 0
    seq = 0
    for i, part in enumerate(parts):
        if part.startswith("EP") and part[2:].isdigit() and i > 0:
            model = parts[i-1]
            ep = int(part[2:])
        if part.startswith("SEQ") and part[3:].isdigit():
            seq = int(part[3:])
    name = path.stem
    step = 0
    if "step" in name:
        try:
            step = int(name.rsplit("step", 1)[1])
        except ValueError:
            step = 0
    return model, ep, seq, step


def process_fixture(args: tuple[str, str, int, int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fixture_path_s, profile_path_s, local_limit_ms, joint_limit_ms = args
    fixture_path = Path(fixture_path_s)
    fixture = load_fixture(fixture_path)
    profile = load_runtime_profile_bundle_json(profile_path_s)
    hardware = profile.transport_profile.hardware_profile
    topology = _default_topology_for_fixture(fixture, topology_id=f"oracle-sweep:{fixture.fixture_id}")
    model, ep, seq, step = _coordinates(fixture_path)
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for window_index in range(len(fixture.windows) - 1):
        current = fixture.windows[window_index]
        following = fixture.windows[window_index + 1]
        p1_matrix = current.payload_matrix("COMBINE")
        p2_matrix = following.payload_matrix("DISPATCH")
        wire = _rscf_wire_cost_model_from_runtime(
            topology=topology,
            hardware_profile=hardware,
            fixture_input=fixture,
            base_layer_index=int(current.layer_id),
            predicted_p2_matrix=tuple(tuple(int(v) for v in row) for row in p2_matrix),
            predicted_p2_confidence_ppm=1_000_000,
            timing_profile=None,
        )
        base = {
            "fixture_path": str(fixture_path),
            "fixture_id": fixture.fixture_id,
            "model": model,
            "ep": ep,
            "sequence_length": seq,
            "step": step,
            "window_index": window_index,
            "anchor_layer_id": int(current.layer_id),
            "target_layer_id": int(following.layer_id),
            "rank_count": int(fixture.world_size),
        }
        p1_tasks = _tasks_from_matrix(
            fixture_id=fixture.fixture_id, window_index=window_index, phase=1, matrix=p1_matrix
        )
        p2_tasks = _tasks_from_matrix(
            fixture_id=fixture.fixture_id, window_index=window_index, phase=2, matrix=p2_matrix
        )
        for phase_name, semantic, tasks in (("P1", 1, p1_tasks), ("P2", 2, p2_tasks)):
            result = solve_exact_wire(
                _problem(fixture.world_size, tasks, tag=f"LOCAL:{phase_name}"),
                wire_cost_model=wire,
                time_limit_ms=local_limit_ms,
                relative_gap=0.02,
                release_mode="PHASE_BARRIER",
                semantic_phase_ordinal=semantic,
            )
            rows.append(_result_row(base, "LOCAL", phase_name, result))
        # Broad Joint sampling: first P12 window of every EP8 fixture.
        if ep == 8 and seq == 128 and window_index == 0:
            joint_tasks = p1_tasks + p2_tasks
            result = solve_exact_wire(
                _problem(fixture.world_size, joint_tasks, tag="JOINT"),
                wire_cost_model=wire,
                time_limit_ms=joint_limit_ms,
                relative_gap=0.02,
                release_mode="RANK_LOCAL",
                semantic_phase_ordinal=None,
            )
            rows.append(_result_row(base, "JOINT", "P12", result))
    return rows, {
        "fixture_path": str(fixture_path),
        "fixture_id": fixture.fixture_id,
        "model": model,
        "ep": ep,
        "sequence_length": seq,
        "step": step,
        "p12_window_count": max(0, len(fixture.windows)-1),
        "elapsed_s": time.monotonic() - started,
        "row_count": len(rows),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["scope"], row["phase"], row["ep"], row["model"], row["sequence_length"])
        buckets.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, group in sorted(buckets.items(), key=lambda item: tuple(map(str, item[0]))):
        gaps = [float(r["gap_pct"]) for r in group if r["gap_pct"] is not None]
        solve = [float(r["solve_time_ms"] or 0.0) for r in group]
        tasks = [int(r["canonical_task_count"] or 0) for r in group]
        out.append({
            "scope": key[0], "phase": key[1], "ep": key[2], "model": key[3], "sequence_length": key[4],
            "subproblem_count": len(group),
            "certified_count": sum(bool(r["certified_optimal"]) for r in group),
            "certified_fraction": sum(bool(r["certified_optimal"]) for r in group) / max(1, len(group)),
            "feasible_count": sum(r["objective_ns"] is not None for r in group),
            "mean_gap_pct": sum(gaps)/len(gaps) if gaps else None,
            "max_gap_pct": max(gaps) if gaps else None,
            "total_solve_s": sum(solve)/1000.0,
            "mean_solve_ms": sum(solve)/len(solve) if solve else None,
            "max_solve_ms": max(solve) if solve else None,
            "mean_tasks": sum(tasks)/len(tasks) if tasks else None,
            "max_tasks": max(tasks) if tasks else None,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-root", type=Path, required=True)
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2)-1)))
    ap.add_argument("--local-limit-ms", type=int, default=1000)
    ap.add_argument("--joint-limit-ms", type=int, default=750)
    ns = ap.parse_args()
    fixtures = sorted(ns.trace_root.rglob("fixtures/*.json"))
    args = [(str(p), str(ns.profile), ns.local_limit_ms, ns.joint_limit_ms) for p in fixtures]
    all_rows: list[dict[str, Any]] = []
    fixture_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    completed = 0
    with ProcessPoolExecutor(max_workers=ns.workers) as pool:
        future_map = {pool.submit(process_fixture, arg): arg[0] for arg in args}
        for future in as_completed(future_map):
            completed += 1
            path = future_map[future]
            try:
                rows, summary = future.result()
                all_rows.extend(rows); fixture_rows.append(summary)
                print(json.dumps({"completed": completed, "total": len(args), "fixture": Path(path).name,
                                  "windows": summary["p12_window_count"], "elapsed_s": round(summary["elapsed_s"],3)}, ensure_ascii=False), flush=True)
            except Exception as exc:
                fixture_rows.append({"fixture_path": path, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
                print(json.dumps({"completed": completed, "total": len(args), "fixture": Path(path).name,
                                  "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), flush=True)
    all_rows.sort(key=lambda r: (r["ep"], r["model"], r["sequence_length"], r["step"], r["window_index"], r["scope"], r["phase"]))
    fixture_rows.sort(key=lambda r: str(r.get("fixture_path", "")))
    summaries = _summaries(all_rows)
    ns.output.mkdir(parents=True, exist_ok=True)
    _write_csv(ns.output / "ORACLE_SUBPROBLEMS.csv", all_rows)
    _write_csv(ns.output / "ORACLE_FIXTURES.csv", fixture_rows)
    _write_csv(ns.output / "ORACLE_GROUPED_SUMMARY.csv", summaries)
    overall = {
        "fixture_count": len(fixtures),
        "completed_fixture_count": sum("error" not in r for r in fixture_rows),
        "failed_fixture_count": sum("error" in r for r in fixture_rows),
        "p12_window_count": sum(int(r.get("p12_window_count", 0)) for r in fixture_rows),
        "local_subproblem_count": sum(r["scope"] == "LOCAL" for r in all_rows),
        "joint_subproblem_count": sum(r["scope"] == "JOINT" for r in all_rows),
        "certified_count": sum(bool(r["certified_optimal"]) for r in all_rows),
        "feasible_count": sum(r["objective_ns"] is not None for r in all_rows),
        "total_solver_s": sum(float(r["solve_time_ms"] or 0.0) for r in all_rows) / 1000.0,
        "wall_s": time.monotonic() - started,
    }
    (ns.output / "ORACLE_OVERALL.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(overall, ensure_ascii=False, indent=2, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()
