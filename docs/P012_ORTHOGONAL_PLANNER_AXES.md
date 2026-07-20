# P012 Orthogonal Planner Axes

The formal P012 runtime represents a strategy as independent axes rather than a
single overloaded `branch` value:

```text
<timing>:<horizon>:<scope>:<engine>:<core>
```

Supported values:

- `timing`: `current`, `future`
- `horizon`: `p012`, `p0123` (`future` currently supports `p012` only)
- `scope`: `local`, `joint`
- `engine`: `event`, `global`
- `core`: `gmwd`, `rsbc`, `rscf`

Examples:

```text
current:p012:local:event:rscf
current:p012:joint:event:rscf
current:p012:local:global:rscf
current:p012:joint:global:rscf
future:p012:local:global:rscf
future:p012:joint:global:rscf
```

## Semantics

- Timing is an outer execution wrapper. `future` runs the same planner before
  the target layer and stores a prepared order for reconciliation.
- Scope changes only the visible ready set and release dependencies.
- Engine changes only how a complete plan is constructed: iterative event
  planning or one-shot global candidate selection.
- Core supplies the shared scoring parameters. The authoritative definitions
  are in `rs.scheduling.families.core.FAMILY_KERNEL_SPECS`.
- Horizon changes advisory information. A local P0123 planner is the strict
  baseline and intentionally cannot consume P3 cross-phase advice.

## Strict comparisons

Joint value:

```text
current:p012:joint:event:rscf
vs current:p012:local:event:rscf
```

Global engine value:

```text
current:p012:joint:global:rscf
vs current:p012:joint:event:rscf
```

Future timing overhead/hiding:

```text
future:p012:joint:global:rscf
vs current:p012:joint:global:rscf
```

Safe selection must pair planners that differ only in scope:

```text
future:p012:joint:global:rscf
vs future:p012:local:global:rscf
```

## Compatibility IDs

Existing three-part IDs remain supported and retain their previous behavior:

- `p012:local:rscf` -> `current:p012:local:event:rscf`
- `p012:event:rscf` -> `current:p012:joint:event:rscf`
- `p012:global:rscf` -> `current:p012:joint:global:rscf`
- `p0123:global:rscf` -> `current:p0123:joint:global:rscf`
- `future_prepared:global:rscf` -> `future:p012:joint:global:rscf`

New experiments should use the explicit five-axis IDs.
