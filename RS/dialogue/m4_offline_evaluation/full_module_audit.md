**Scope**

Read modules:

- `src/rs/runtime/offline/**`
- `src/rs/scheduling/multiphase/**`
- `src/rs/scheduling/reference/**`
- `src/rs/runtime/online/megatron_ep/async_release/simulator.py`
- `src/rs/prediction/**`
- `src/rs/planning/**`
- `experiments/offline/**`
- `tests/offline/**`
- `tests/contract/*offline*`
- `tests/solver/**`

**Primary findings**

1. Offline planning input was still split across:
   - `runtime/offline/replay_unified.py`
   - `runtime/offline/runner.py`
   - `experiments/run_offline_replay.py`
   - multiple experiment-local `_build_problem*` helpers
2. Formal `PlanningRequest` already exists and is reusable for offline.
3. Existing replay path still builds `MultiPhaseSchedulingProblem` and legacy forecast objects.
4. Existing exact reference coverage is split between:
   - `scheduling/reference/exact_small_instance.py`
   - `scheduling/reference/oracle_guided.py`
   - experiment-local OR-Tools CP-SAT logic
5. Existing simulation coverage is split between:
   - `runtime/online/megatron_ep/async_release/simulator.py`
   - `scheduling/multiphase/streaming_simulator.py`
6. Truth/hint isolation was already partially repaired in M0, but offline contracts were still implicit.

**File classification**

- `src/rs/runtime/offline/replay_unified.py`: `MIGRATE`
- `src/rs/runtime/offline/runner.py`: `THIN_WRAPPER`
- `src/rs/runtime/offline/prediction/evaluation.py`: `REFERENCE_ONLY`
- `src/rs/runtime/offline/prediction/*`: `REFERENCE_ONLY`
- `src/rs/scheduling/reference/exact_small_instance.py`: `REFERENCE_ONLY`
- `src/rs/scheduling/reference/oracle_guided.py`: `REFERENCE_ONLY`
- `src/rs/runtime/online/megatron_ep/async_release/simulator.py`: `SIMULATION_ONLY`
- `experiments/run_offline_replay.py`: `THIN_WRAPPER`
- `experiments/offline/run_prediction_oracle_baseline_closure.py`: `REFERENCE_ONLY`

**New formal surface introduced in M4 Phase A**

- `src/rs/core/contracts/offline.py`
- `src/rs/offline/**`
- `src/rs/simulation/**`

**Phase B dependency**

Offline/online materialization parity and execution-semantics parity still depend on M123 integration becoming genuinely READY. This branch does not modify M1/M2/M3 exclusive files.
