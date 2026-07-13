# M1 Preparation Flow

## Submission

- caller uses `TargetLayerPlannerService.submit(...)`
- result is one of:
  - `ACCEPTED`
  - `REPLACED_STALE`
  - `DROPPED_OVERLOAD`
  - `REJECTED_EXPIRED`
  - `REJECTED_CLOSED`

Submission is non-blocking and uses `put_nowait`.

## Worker

Worker responsibilities are local only:

1. predictor
2. formal planner build / selector
3. build ready publication

Worker does not directly publish into store agreement path anymore.

## Publication

Main thread pumps:

1. `drain_ready_publications()`
2. `publish_ready_plan(...)`
3. `publish_agreed_plan(...)`

This keeps collective / publish ownership out of the worker.
