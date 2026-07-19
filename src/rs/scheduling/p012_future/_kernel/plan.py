from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

import numpy as np

from .contracts import _digest, _freeze_metadata, semantic_metadata

PHASE_NAMES = ("p0_dispatch", "p1_return", "p2_next_dispatch")


@dataclass(frozen=True)
class PlannedFlow:
    parent_flow_id: str
    segment_id: str
    phase: str
    src_rank: int
    dst_rank: int
    row_offset: int
    row_count: int
    release_state: str
    executable: bool

    def to_dict(self) -> dict:
        return {
            "parent_flow_id": self.parent_flow_id, "segment_id": self.segment_id,
            "phase": self.phase, "src_rank": int(self.src_rank), "dst_rank": int(self.dst_rank),
            "row_offset": int(self.row_offset), "row_count": int(self.row_count),
            "release_state": self.release_state, "executable": bool(self.executable),
        }


@dataclass(frozen=True)
class PlanWave:
    wave_id: int
    start_time: float
    end_time: float
    estimated_duration: float
    logical_quantum: int
    flows: tuple[PlannedFlow, ...]

    def to_dict(self) -> dict:
        return {
            "wave_id": int(self.wave_id), "start_time": float(self.start_time),
            "end_time": float(self.end_time), "estimated_duration": float(self.estimated_duration),
            "logical_quantum": int(self.logical_quantum), "flows": [flow.to_dict() for flow in self.flows],
        }


@dataclass(frozen=True)
class WindowPlan:
    """Materialized audit plan. Runtime scheduling uses CompactWindowPlan."""
    planner_id: str
    planner_family: str
    branch: str
    request_digest: str
    forecast: bool
    makespan: float
    waves: tuple[PlanWave, ...]
    phase_completion: tuple[float, float, float]
    release1: tuple[float, ...]
    release2: tuple[float, ...]
    valid: bool
    metadata: dict = field(default_factory=dict)
    _digest_cache: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.planner_id or not self.planner_family or not self.branch:
            raise ValueError("plan identity fields must be non-empty")
        if len(str(self.request_digest)) != 64:
            raise ValueError("request_digest must be a SHA-256 digest")
        if not np.isfinite(float(self.makespan)) or float(self.makespan) < 0:
            raise ValueError("makespan must be finite and non-negative")
        if len(self.phase_completion) != 3:
            raise ValueError("phase_completion must contain P0/P1/P2 completion times")
        if len(self.release1) != len(self.release2):
            raise ValueError("release1/release2 world sizes must match")
        object.__setattr__(self, "waves", tuple(self.waves))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def semantic_payload(self) -> dict:
        return {
            "plan_semantic_version": "window_plan_v5_materialized",
            "planner_id": self.planner_id, "planner_family": self.planner_family,
            "branch": self.branch, "request_digest": self.request_digest,
            "forecast": bool(self.forecast), "makespan": float(self.makespan),
            "waves": [wave.to_dict() for wave in self.waves],
            "phase_completion": list(map(float, self.phase_completion)),
            "release1": list(map(float, self.release1)), "release2": list(map(float, self.release2)),
            "valid": bool(self.valid), "metadata": semantic_metadata(self.metadata),
        }

    def semantic_digest(self) -> str:
        cached = self._digest_cache
        if cached is None:
            cached = _digest(self.semantic_payload()); object.__setattr__(self, "_digest_cache", cached)
        return cached

    def to_dict(self) -> dict:
        payload = self.semantic_payload(); payload["plan_digest"] = self.semantic_digest(); return payload


@dataclass(frozen=True)
class CompactWindowPlan:
    """Primary offline/online plan contract backed by compact arrays.

    The executor/compiler consumes these arrays directly. Human-readable flow
    objects are materialized lazily and are not charged to planner latency.
    """
    planner_id: str
    planner_family: str
    branch: str
    request_digest: str
    forecast: bool
    result: tuple
    metadata: dict = field(default_factory=dict)
    _digest_cache: str | None = field(default=None, init=False, repr=False, compare=False)
    _materialized_cache: WindowPlan | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.result) != 13:
            raise ValueError("compact plan result must have 13 fields")
        if not self.planner_id or not self.planner_family or not self.branch:
            raise ValueError("plan identity fields must be non-empty")
        if len(str(self.request_digest)) != 64:
            raise ValueError("request_digest must be a SHA-256 digest")
        if int(self.result[1]) < 0:
            raise ValueError("wave_count must be non-negative")
        if not np.isfinite(float(self.result[0])) or float(self.result[0]) < 0:
            raise ValueError("makespan must be finite and non-negative")
        frozen: list[Any] = []
        for value in self.result:
            if isinstance(value, np.ndarray):
                # Prepared runtime binders may hand over ownership of already
                # contiguous, read-only arrays.  Reuse them instead of copying
                # the complete execution plan on the target critical path.
                if value.flags.c_contiguous and not value.flags.writeable:
                    arr = value
                else:
                    arr = np.ascontiguousarray(value).copy()
                    arr.setflags(write=False)
                frozen.append(arr)
            else:
                frozen.append(value)
        object.__setattr__(self, "result", tuple(frozen))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        phase, dst, size = self.result[2], self.result[3], self.result[4]
        if not all(isinstance(value, np.ndarray) for value in self.result[2:12]):
            raise ValueError("compact plan fields 2..11 must be numpy arrays")
        if phase.ndim != 2 or dst.shape != phase.shape or size.shape != phase.shape:
            raise ValueError("phase/dst/size must be equally shaped [wave, source] arrays")
        waves, world = phase.shape
        if waves != int(self.result[1]):
            raise ValueError("wave_count does not match phase array")
        for name, value in (("quantum", self.result[5]), ("duration", self.result[6]),
                            ("starts", self.result[7]), ("ends", self.result[8])):
            if value.shape != (waves,):
                raise ValueError(f"{name} must have shape [wave_count]")
        if self.result[9].shape != (3,):
            raise ValueError("phase_completion must have shape [3]")
        if self.result[10].shape != (world,) or self.result[11].shape != (world,):
            raise ValueError("release arrays must have shape [world_size]")

    def structural_errors(self, *, atol: float = 1e-8) -> tuple[str, ...]:
        """Return semantic/shape errors without invoking scheduling logic."""
        errors: list[str] = []
        phase = np.asarray(self.phase); dst = np.asarray(self.dst); size = np.asarray(self.size)
        duration = np.asarray(self.duration); starts = np.asarray(self.starts); ends = np.asarray(self.ends)
        if not all(np.isfinite(value).all() for value in (duration, starts, ends, np.asarray(self.result[9]), np.asarray(self.result[10]), np.asarray(self.result[11]))):
            errors.append("compact plan timing/release arrays must be finite")
        if np.any(duration < 0) or np.any(starts < 0) or np.any(ends < 0):
            errors.append("compact plan times must be non-negative")
        active = phase >= 0
        if np.any(active & (phase > 2)):
            errors.append("compact plan contains phase outside P0/P1/P2")
        if np.any(active & ((dst < 0) | (dst >= phase.shape[1]))):
            errors.append("compact plan contains destination outside world")
        if np.any(active & (size <= 0)):
            errors.append("active compact flows must have positive row_count")
        if np.any((~active) & ((dst >= 0) | (size != 0))):
            errors.append("inactive compact slots must use dst=-1 and size=0")
        for wave in range(self.wave_count):
            sources = np.flatnonzero(active[wave])
            destinations = dst[wave, sources]
            if len(set(int(x) for x in destinations)) != len(destinations):
                errors.append(f"wave {wave} receives more than once at a destination")
            if any(int(source) == int(destination) for source, destination in zip(sources, destinations, strict=True)):
                errors.append(f"wave {wave} contains a local/self transfer")
            if abs(float(ends[wave] - starts[wave] - duration[wave])) > float(atol):
                errors.append(f"wave {wave} timestamps disagree with duration")
            if wave and float(starts[wave]) + float(atol) < float(ends[wave - 1]):
                errors.append(f"wave {wave} overlaps previous wave")
        final_end = float(ends[-1]) if self.wave_count else 0.0
        if abs(float(self.makespan) - final_end) > float(atol):
            errors.append("makespan disagrees with final wave end")
        return tuple(errors)

    @property
    def makespan(self) -> float: return float(self.result[0])
    @property
    def wave_count(self) -> int: return int(self.result[1])
    @property
    def phase(self) -> np.ndarray: return self.result[2]
    @property
    def dst(self) -> np.ndarray: return self.result[3]
    @property
    def size(self) -> np.ndarray: return self.result[4]
    @property
    def quantum(self) -> np.ndarray: return self.result[5]
    @property
    def duration(self) -> np.ndarray: return self.result[6]
    @property
    def starts(self) -> np.ndarray: return self.result[7]
    @property
    def ends(self) -> np.ndarray: return self.result[8]
    @property
    def phase_completion(self) -> tuple[float, float, float]: return tuple(float(x) for x in self.result[9])
    @property
    def release1(self) -> tuple[float, ...]: return tuple(float(x) for x in self.result[10])
    @property
    def release2(self) -> tuple[float, ...]: return tuple(float(x) for x in self.result[11])
    @property
    def valid(self) -> bool: return bool(self.result[12])

    @property
    def waves(self) -> tuple[PlanWave, ...]:
        return self.materialize().waves

    def materialize(self) -> WindowPlan:
        cached = self._materialized_cache
        if cached is None:
            cached = tuple_to_plan(
                self.result, planner_id=self.planner_id, planner_family=self.planner_family,
                branch=self.branch, request_digest=self.request_digest,
                forecast=self.forecast, metadata=self.metadata,
            )
            object.__setattr__(self, "_materialized_cache", cached)
        return cached

    def semantic_digest(self) -> str:
        cached = self._digest_cache
        if cached is not None:
            return cached
        h = hashlib.sha256()
        header = {
            "plan_semantic_version": "compact_window_plan_v1",
            "planner_id": self.planner_id, "planner_family": self.planner_family,
            "branch": self.branch, "request_digest": self.request_digest,
            "forecast": bool(self.forecast), "makespan": self.makespan,
            "wave_count": self.wave_count, "valid": self.valid,
            "metadata": semantic_metadata(self.metadata),
        }
        h.update(json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode())
        for arr in self.result[2:12]:
            a = np.ascontiguousarray(arr)
            h.update(str(a.dtype).encode()); h.update(np.asarray(a.shape, dtype=np.int64).tobytes()); h.update(a.tobytes())
        cached = h.hexdigest(); object.__setattr__(self, "_digest_cache", cached); return cached

    def to_dict(self, *, materialize: bool = False) -> dict:
        if materialize:
            return self.materialize().to_dict()
        return {
            "plan_semantic_version": "compact_window_plan_v1", "planner_id": self.planner_id,
            "planner_family": self.planner_family, "branch": self.branch,
            "request_digest": self.request_digest, "forecast": self.forecast,
            "makespan": self.makespan, "wave_count": self.wave_count,
            "phase": self.phase.tolist(), "dst": self.dst.tolist(), "size": self.size.tolist(),
            "quantum": self.quantum.tolist(), "duration": self.duration.tolist(),
            "starts": self.starts.tolist(), "ends": self.ends.tolist(),
            "phase_completion": list(self.phase_completion), "release1": list(self.release1),
            "release2": list(self.release2), "valid": self.valid,
            "metadata": semantic_metadata(self.metadata), "plan_digest": self.semantic_digest(),
        }


def tuple_to_plan(result, *, planner_id: str, planner_family: str, branch: str,
                  request_digest: str, forecast: bool, metadata: dict | None = None) -> WindowPlan:
    makespan, wave_count, phase, dst, size, quantum, duration, starts, ends, phase_done, rel1, rel2, valid = result
    offsets: dict[tuple[int, int, int], int] = {}; waves: list[PlanWave] = []
    for wave_id in range(int(wave_count)):
        flows: list[PlannedFlow] = []
        for source in range(phase.shape[1]):
            p = int(phase[wave_id, source]); target = int(dst[wave_id, source]); rows = int(size[wave_id, source])
            if p < 0 or target < 0 or rows <= 0: continue
            key=(p,source,target); offset=offsets.get(key,0); parent=f"{PHASE_NAMES[p]}:{source}->{target}"
            advisory=bool(forecast and p==2)
            flows.append(PlannedFlow(parent,f"{parent}:segment{offset}","p2_next_dispatch_forecast" if advisory else PHASE_NAMES[p],
                                     source,target,offset,rows,"advisory_only" if advisory else ("ready" if p==0 else "barrier_released"),not advisory))
            offsets[key]=offset+rows
        waves.append(PlanWave(wave_id,float(starts[wave_id]),float(ends[wave_id]),float(duration[wave_id]),int(quantum[wave_id]),tuple(flows)))
    return WindowPlan(str(planner_id),str(planner_family),str(branch),str(request_digest),bool(forecast),float(makespan),tuple(waves),
                      tuple(float(x) for x in phase_done),tuple(float(x) for x in rel1),tuple(float(x) for x in rel2),bool(valid),dict(metadata or {}))


def tuple_to_compact_plan(result, *, planner_id: str, planner_family: str, branch: str,
                          request_digest: str, forecast: bool, metadata: dict | None = None,
                          trusted_arrays: bool = False) -> CompactWindowPlan:
    values = list(result)
    if trusted_arrays:
        for value in values:
            if isinstance(value, np.ndarray):
                if not value.flags.c_contiguous:
                    raise ValueError("trusted compact-plan arrays must be C-contiguous")
                value.setflags(write=False)
    return CompactWindowPlan(str(planner_id),str(planner_family),str(branch),str(request_digest),bool(forecast),tuple(values),dict(metadata or {}))


def plan_to_compact(plan: CompactWindowPlan | WindowPlan) -> dict:
    if isinstance(plan, CompactWindowPlan):
        return {"phase": plan.phase, "dst": plan.dst, "size": plan.size}
    n=0
    for wave in plan.waves:
        for flow in wave.flows:n=max(n,flow.src_rank+1,flow.dst_rank+1)
    phase=np.full((len(plan.waves),n),-1,np.int8);dst=np.full((len(plan.waves),n),-1,np.int16);size=np.zeros((len(plan.waves),n),np.int32)
    phase_map={"p0_dispatch":0,"p1_return":1,"p2_next_dispatch":2,"p2_next_dispatch_forecast":2}
    for wave in plan.waves:
        for flow in wave.flows:
            phase[wave.wave_id,flow.src_rank]=phase_map[flow.phase];dst[wave.wave_id,flow.src_rank]=flow.dst_rank;size[wave.wave_id,flow.src_rank]=flow.row_count
    return {"phase":phase,"dst":dst,"size":size}


__all__=["CompactWindowPlan","PlanWave","PlannedFlow","WindowPlan","plan_to_compact","tuple_to_compact_plan","tuple_to_plan"]
