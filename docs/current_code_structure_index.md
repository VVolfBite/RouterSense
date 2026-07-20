# Current code structure index

- `rs.core.contracts`: typed experiment, result and performance contracts;
- `rs.planning`: canonical planner registry and Current/Future wrappers;
- `rs.scheduling.p012_future._kernel`: shared RSCF/RSBC/GMWD planner kernel;
- `rs.scheduling.phase_local`: deployable FIFO/Greedy/Birkhoff controls;
- `rs.reference.baselines`: FAST/Aurora/iSLIP and other offline references;
- `rs.evaluation`: metric derivation and comparison helpers;
- `rs.runtime.online.megatron_ep`: observation, planning, materialization,
  execution, measurement and evidence chain.

Large runtime-file splitting is deferred to round 2 after this semantic
convergence checkpoint is frozen.
