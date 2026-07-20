# RouterSense All-Ready Reconstruction Provenance

This tree was reconstructed from the user-supplied Round 2 checkpoint after the
original final deployment ZIP was no longer available.

## Inputs

- Round 2 checkpoint SHA-256: `37c1aaf03c1ded6d9b2c431be1ff2bc72d424a236f257cb1aefeb015130a2462`
- Round 1 comparison checkpoint SHA-256: `bd67ea7ab8440da7e7116bdc0d59c924594a71239ab14ba8a62441a9cf9f9926`
- Historical target release identity from the retained final report:
  `d620a39200d8c68ced27ef999ccc2cc94c95c066`

The historical target identity is a reconstruction specification, not the Git
identity of this rebuilt tree. The authoritative identity of each rebuilt
archive is the commit and tree digest in its root `source_manifest.json`.

## Reapplied final-round closure

- process-once attach-time warmup for Event/Global Future-P012 planning and both binding paths;
- removal of deprecated registry shims after confirming zero active references;
- v2 all-ready gate with an isolated empty Numba cache stage;
- source archive filtering for Python/Numba caches and the Round 1 removed-source snapshot;
- replay truth/hint isolation: forecast plans remain advisory, realized P2 traffic is bound only after the planning digest is frozen, and execution-bound plans carry a separate digest;
- preservation of the deployment pipeline, official strategy configuration, formal metrics, and same-engine safe fallback boundaries.

## Evidence boundary

The rebuilt source is validated on CPU/Gloo and deployment dry-run paths. The
historical trace reports may be carried in the release bundle as retained
reference evidence, but they are not relabeled as a fresh rerun. CUDA/NCCL and
physical two-node execution must still be run on the target cluster.
