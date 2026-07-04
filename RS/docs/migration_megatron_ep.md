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

## Current Blockers

- `megatron-core` missing in local environment
- `megatron-bridge` missing in local environment
- `transformer-engine` missing in local environment
- local `OLMoE-1B-7B-0125` checkpoint missing

## Immediate Next Command

```bash
python integrations/megatron_ep/verify_env.py \
  --model /root/autodl-tmp/models/OLMoE-1B-7B-0125
```
