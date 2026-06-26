# POC2 Single Node Scheduler Prototype

This stage upgrades proxy analysis into a runnable single-node 4-GPU scheduler prototype.

It is not a final EP runtime. It uses real routing extraction when GPUs are available, then drives bucket release and per-rank service order inside one unified scheduler interface.
