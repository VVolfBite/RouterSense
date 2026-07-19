# Scheduling Theory Notes

## Formal Exact Oracle Model

The formal RouterSense oracle comparison is a scope-only comparison under one
canonical discrete runtime model. Both `O_local` and `O_joint` use:

- canonical remote-edge bucket tasks;
- the same configured `bucket_rows`;
- full-duplex conflict-free matching waves;
- wave duration equal to the maximum row count in the wave;
- rank-local P0→P1→P2 release semantics;
- the same replay objective and deterministic tie-breaking.

The only difference is scope:

- `O_local` solves P0, P1 and P2 exactly but independently, with phase barriers;
- `O_joint` solves the same bucket tasks exactly in a dependency-released joint
  search space.

The current reference model ID is
`routersense_exact_bucket_wave_release_v2`. The certified implementation is
restricted to tiny instances (at most 4 ranks and 12 canonical bucket tasks).
Unsupported larger instances fail closed and must not be described as exact.

## Birkhoff / BvN as a Fluid Sensitivity Reference

Birkhoff-von Neumann decomposition remains useful for a narrow single-phase
fluid/crossbar model. It is deterministic and gives an oracle-like load
reference under permutation-slot service semantics. It does not share the
discrete bucket/wave model used by the formal exact pair, and therefore is not
the formal `O_local` in the paper optimality-gap table.

`B_birkhoff` / `birkhoff_phase_local` are strong engineering baselines.
`birkhoff_von_neumann_fluid` is a separate fluid sensitivity reference. Neither
should be substituted for the exact bucket-wave `O_local`.

### Caveat

BvN does not necessarily minimize wave count, launch/setup cost, or a
dependency-released multi-stage objective. Results from the fluid reference and
the exact bucket-wave model must be reported in separate tables.

### Citation placeholders

- TODO[BibTeX]: Birkhoff-von Neumann theorem / matrix decomposition reference
- TODO[BibTeX]: crossbar scheduling / input-queued switch decomposition reference

## Why Joint / Multiphase Scheduling Is Harder

RouterSense P0/P1/P2 scheduling is not just “do Birkhoff three times”.

The coupled problem includes:

- multi-stage communication
- release dependencies between phases
- downstream pressure from future communication
- rank-level barrier and completion sensitivity
- optional prediction inputs for future traffic

This makes the problem closer to:

- dependency-aware coflow scheduling
- multi-stage communication scheduling
- open-shop-like / job-shop-like coupled scheduling

Under those semantics, the problem is substantially harder than single-phase BvN,
and exact or near-exact joint references should be solver-based or iterative joint
references rather than BvN.

For RouterSense, the correct theoretical split is:

- `O_local_phase_oracle`: exact canonical bucket-wave model with local scope;
- `O_joint` / compatibility alias `O_joint_cp_sat_oracle`: the same exact model with joint scope;
- `birkhoff_von_neumann_fluid`: a separate phase-local fluid sensitivity reference.

### Citation placeholders

- TODO[BibTeX]: dependency coflow / coflow scheduling NP-hardness
- TODO[BibTeX]: open-shop / multi-stage scheduling hardness reference
- TODO[BibTeX]: exact dependency-aware scheduling / dynamic programming formulation reference

## RouterSense Classification Rules

- `B_*` means phase-local / independent-phase version of a heuristic family.
- `U_*` means RouterSense joint / multiphase version of the same family.
- `O_*` means theoretical reference, not a regular RouterSense scheduler family.
- `*_wave` and `*_atomic` are legacy granularity variants, not distinct algorithm
  families.
- `routersense_p0p1p2_hint` is an early online adapter, not the core POC1 U-family.
