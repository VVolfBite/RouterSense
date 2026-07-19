# Legacy HF EP Harness

- `pipeline = legacy`
- `status = frozen`
- `not_for_formal_ep_runtime = true`
- `not_for_performance_claims = true`

## Archive

- archive tag: `legacy-hf-ep-harness-20260704T100531Z`
- commit SHA: `6ca2fc8`
- bundle filename: `archives/routersense-legacy-hf-ep-harness-20260704T100531Z.bundle`
- bundle SHA256:
  `1df67c032114be95ab69785e02103b993f4820ebb65675fd47b3c24f058ce41d`
- archive date: `2026-07-04T10:05:31Z`

## What This Legacy Harness Implements

- HuggingFace-based WS=2 MoE-layer observation harness
- local/remote route partition and metadata agreement
- variable-size hidden+metadata all-to-all dispatch
- owner-rank expert compute on extracted local expert weights
- inverse combine all-to-all and weighted scatter
- distributed numerical parity against captured single-rank layer output
- route/manifest/trace schema artifacts for offline analysis

## What It Does Not Implement

- real Megatron Core host runtime
- true expert-sharded checkpoint residency
- multi-layer EP forward replacement
- production serving runtime
- formal performance baseline for native EP systems
- RouterSense schedule injection into a real host runtime

## Usage Restrictions

- Do not use legacy throughput or latency numbers as formal EP runtime results.
- Do not claim this harness is a production DEP runtime.
- Do not connect new formal runtime code to legacy runtime objects.
- New integrations must communicate with legacy only through serialized artifacts.
