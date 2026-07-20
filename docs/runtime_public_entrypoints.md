# Runtime public entrypoints

Supported public strategies are native/disabled, FIFO, Greedy, Birkhoff
phase-local and explicit orthogonal RouterSense strategy IDs. Prepared plans
are materialized by `prepared_priority` after a formal planner ID has been
selected. Offline references and exact solvers are rejected by the runtime
wire.
