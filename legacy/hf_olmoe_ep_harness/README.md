# Legacy HF OLMoE EP Harness

This directory marks the historical HuggingFace/self-built EP harness line.

- It is retained only for historical POC reference, interface archaeology, and
  old-result reproduction.
- Formal runtime code must not import from this legacy line.
- Formal evaluation and paper claims must not treat this line as real Megatron
  EP execution evidence.

The actual historical implementation still lives in:

- `src/rs/online/olmoe_ep/`
- `src/rs/runtime/distributed_ep/`
- `experiments/distributed/`
- `experiments/online/bench_native_ep.py`
- `experiments/online/bench_scheduled_ep.py`

Round-1 cleanup keeps those paths in place but treats them as legacy-only.
