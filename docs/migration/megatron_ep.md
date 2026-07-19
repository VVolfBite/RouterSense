# Megatron EP Migration

## Decision

RouterSense formal DEP runtime work is migrating from the frozen HuggingFace
WS=2 layer harness to:

```text
Megatron Bridge
-> Megatron Core native MoE EP
-> NCCL all-to-all baseline
-> RouterSense read-only observer
-> RouterSense no-op dispatcher facade
```

## Legacy Boundary

The old line is frozen in `legacy/README.md` and tagged as:

- `legacy-hf-ep-harness-20260704T100531Z`

It remains valid only for:

- schema regression
- route semantics tests
- offline scheduler prototyping
- appendix-level implementation history

It is not valid for:

- formal EP runtime claims
- formal performance claims
- native Megatron dispatch/combine measurements

## New Gates

- Gate A: native Megatron EP=2 all-to-all smoke
- Gate B: native EP trace export with real remote dispatch/combine
- Gate C: RouterSense no-op facade numerical equivalence

Before A/B/C pass, do not implement scheduled transport injection.

## Current Status

The migration line has already passed the first three runtime gates on a local
single-node, 2xGPU setup using:

- backend: `nccl`
- EP size: `2`
- dispatcher: native `alltoall`
- checkpoint: local `OLMoE-1B-7B-0924`

Validated:

- Gate A: native Megatron EP forward with real remote dispatch/combine
- Gate B: lightweight RouterSense observer trace export
- Gate C: `native_order` no-op facade numerical equivalence

Current default local checkpoint:

- `/root/autodl-tmp/models/OLMoE-1B-7B-0924`

Remaining blocker:

- if we later require a different OLMoE revision, that checkpoint still needs
  to be provisioned explicitly

## Immediate Next Command

```bash
python integrations/megatron_ep/verify_env.py \
  --model /root/autodl-tmp/models/OLMoE-1B-7B-0924
```
