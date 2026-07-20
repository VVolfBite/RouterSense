# Phase 0C Distributed EP Contract

Phase 0C targets a real multi-node NCCL bring-up for OLMoE inference. The concrete rank count is inventory-driven; the immediate working target in the current PPIO environment is 2 nodes x 1 GPU, 2 ranks total. Both nodes are workers; node0 only serves as the rendezvous endpoint.

Deployment prepares local model caches on both nodes. Inference does not move checkpoints or expert weights across nodes. Cross-node traffic is limited to token hidden states, route metadata, and expert outputs.

This phase does not implement RouteSense scheduling, oracle routing, or benchmark comparisons. It is a correctness bring-up for distributed expert execution only.
