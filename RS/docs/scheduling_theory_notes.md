# Scheduling Theory Notes

## Birkhoff / BvN as Phase-Local Oracle-Like Reference

RouterSense treats Birkhoff-von Neumann style decomposition as a special reference,
not as just another heuristic.

Core reasoning:

1. The Birkhoff-von Neumann theorem states that a doubly stochastic matrix can be
   decomposed into a convex combination of permutation matrices.
2. In a single-phase crossbar / all-to-all fluid service model, each permutation
   matrix corresponds to one conflict-free matching slot.
3. After normalizing a single-phase traffic matrix into the appropriate stochastic
   form, BvN decomposition gives a deterministic phase-local schedule construction.

Because of that, under the narrow semantic scope below, Birkhoff behaves like a
phase-local oracle-like deterministic reference:

- single phase only
- fluid or permutation-slot service model
- crossbar / conflict-free matching semantics
- objective focused on phase-local communication makespan
- no kernel launch, setup, or wave-count overhead in the objective

This is why RouterSense uses the fluid `birkhoff_von_neumann_fluid` reference as
the formal `O_local_phase_oracle`, while `B_birkhoff` remains the strong
engineering phase-local baseline rather than the oracle object itself.
as a generic baseline heuristic.

### Caveat

BvN does **not** automatically optimize every engineering metric:

- it does not necessarily minimize permutation count
- it does not necessarily minimize wave count
- it does not necessarily minimize launch/setup overhead

Therefore, when the evaluation objective includes wave count, kernel launch count,
or fixed communication setup cost, BvN must be described as:

- phase-local oracle-like reference for fluid makespan semantics

and not as the oracle for every system metric.

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

- `O_local_phase_oracle`: Birkhoff-like local deterministic reference
- `O_joint_CT_oracle` / `O_joint_cp_sat_oracle`: joint solver-based reference

### Citation placeholders

- TODO[BibTeX]: dependency coflow / coflow scheduling NP-hardness
- TODO[BibTeX]: open-shop / multi-stage scheduling hardness reference
- TODO[BibTeX]: CP-SAT / exact scheduling formulation reference

## RouterSense Classification Rules

- `B_*` means phase-local / independent-phase version of a heuristic family.
- `U_*` means RouterSense joint / multiphase version of the same family.
- `O_*` means theoretical reference, not a regular RouterSense scheduler family.
- `*_wave` and `*_atomic` are legacy granularity variants, not distinct algorithm
  families.
- `routersense_p0p1p2_hint` is an early online adapter, not the core POC1 U-family.
