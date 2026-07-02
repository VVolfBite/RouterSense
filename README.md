# RouteSense

RouteSense is a distributed MoE scheduling and deployment project. The formal
mainline lives under `RS/`. Historical POC material is preserved in `legacy/`
and curated result snapshots live under `archive/backup/`.

## Mainline

- `RS/src/routesense/`
  Mainline runtime, scheduler, trace, topology, oracle, and evaluation code.
- `RS/deploy/`
  Inventory, dry-run launch contracts, and deployment helpers.
- `RS/experiments/`
  Deployment smokes, distributed bring-up, and scheduler evaluation entrypoints.
- `RS/docs/`
  Mainline technical docs and current design notes.
- `RS/tests/`
  Mainline regression tests.
- `RS/artifacts/`
  Active working outputs only. This tree should stay trimmed.

## Curated Backup

- `archive/backup/`
  Git-controlled curated backup area.
- `archive/backup/README.md`
  Index of the currently retained result snapshots.
- `archive/backup/docs/`
  Archived copies of root-level planning / task / elimination documents.

As of the current cleanup, only three experiment backup groups are intentionally
retained there:

1. cross-layer prediction validity
2. oracle / fast optimization-gap study
3. execution-window multiscale scheduler study

## Legacy

- `legacy/poc1/`
  Router observability and proxy-era experiments.
- `legacy/poc2/`
  Single-node NCCL harness and historical scheduler diagnostics.
- `legacy/shared/`
  Shared historical docs and test assets.

The legacy trees are kept for reference and reproducibility. They are not part
of the formal `RS/` runtime path, and the mainline tests explicitly guard
against accidental legacy imports.

## Operator Channel

- `rs_channel/instruction/`
  Incoming operator instructions synchronized through git.
- `rs_channel/reply/`
  Outgoing step reports.

## Current Phase

The current mainline focus is the path from offline scheduler validation to
real multi-GPU MoE communication execution under the `RS/` stack. Historical
POC documents remain available for context, but they are not the source of
truth for the deployment mainline.

## Current POC-Line1 Presets

- Standard non-repeated prompt corpus:
  `RS/artifacts/poc_line1/prompt_sets/olmoe_oasst256_unique.jsonl`
- Tiered scheduler candidate presets:
  `RS/experiments/poc_line1/configs/candidate_tiers.json`

Example:

```bash
cd /root/autodl-tmp/RouterSense/RS
OMP_NUM_THREADS=1 PYTHONPATH=src python -u experiments/poc_line1/exp_pairwise_candidate_compare.py \
  --config-json experiments/poc_line1/configs/candidate_tiers.json \
  --config-key tier2 \
  --trace-jsonl artifacts/poc_line1/full_sequence_trace_olmoe_mix200_unique_v1/trace.jsonl \
  --hidden-states-path artifacts/poc_line1/full_sequence_trace_olmoe_mix200_unique_v1/hidden_states.pt \
  --gate-weights-path artifacts/poc_line1/full_sequence_trace_olmoe_mix200_unique_v1/gate_weights.pt \
  --output-dir artifacts/poc_line1/tier2_compare
```
