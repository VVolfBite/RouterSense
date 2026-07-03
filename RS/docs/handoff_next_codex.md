# RouteSense Handoff For Next Codex

## 1. Project Snapshot

RouteSense is a MoE communication scheduling and deployment project. The work is split into two tightly related lines:

1. offline scheduler research on trace-derived traffic matrices
2. real distributed EP execution bring-up on top of NCCL collectives

The repository root is `/root/autodl-tmp/RouterSense`. The formal mainline is under `RS/`.

The current project claim is not "better routing weights." The main claim is:

- MoE communication is highly skewed.
- Cross-layer future demand is partially predictable.
- There is a measurable optimization gap between simple phase-local schedules and stronger joint / prediction-aware schedules.
- The next deployment target is to inject these schedules into real EP communication.

## 2. Main Objective

There are two near-term deliverables:

1. finish a reproducible offline evaluation story for scheduler candidates
2. finish an end-to-end real execution chain:
   model load -> trace -> dispatch plan -> scheduler -> NCCL communication -> local expert compute -> correctness check

The deployment line is currently a correctness bring-up, not a production performance claim.

## 3. Model Pool

The intended four-model comparison pool is:

1. `allenai/OLMoE-1B-7B-0924`
2. `Qwen/Qwen1.5-MoE-A2.7B`
3. `mistralai/Mixtral-8x7B-Instruct-v0.1`
4. `DeepSeek-V2-Lite` family

Models confirmed on the current machine:

1. `/root/autodl-tmp/models/OLMoE-1B-7B-0924`
2. `/root/autodl-tmp/models/Qwen1.5-MoE-A2.7B`
3. `/root/autodl-tmp/models/Mixtral-8x7B-Instruct-v0.1`

`DeepSeek-V2-Lite` was planned but is not currently present in `/root/autodl-tmp/models/`.

Current real deployment work is only wired for OLMoE. Qwen trace support exists, but real distributed adapter parity is not yet complete.

## 4. Code Map

Key directories:

- `RS/src/routesense/scheduler/`
  Offline scheduling algorithms, oracle, strategy registry, global matching prototypes.
- `RS/src/routesense/runtime/distributed_ep/core/`
  Manifest, placement, scheduler facade, wave planner, wave executor, NCCL/P2P execution helpers.
- `RS/src/routesense/runtime/distributed_ep/adapter/`
  Model-specific bridge layer. OLMoE is the primary implemented adapter.
- `RS/src/routesense/trace/`
  Trace extraction and trace schema handling.
- `RS/src/routesense/evaluation/`
  Offline pairwise analysis and summary logic.
- `RS/experiments/poc_line1/`
  Offline trace collection and scheduler experiments.
- `RS/experiments/distributed/`
  Real execution smokes and distributed bring-up entrypoints.
- `RS/docs/`
  Current mainline documentation.
- `archive/backup/`
  Curated, git-tracked result snapshots from earlier milestones.

## 5. Scheduler Taxonomy

Current naming convention:

- `B_` prefix:
  baseline / phase-local methods that do not use future-phase values to guide earlier-phase decisions
- `U_` prefix:
  unified / our side, meaning the algorithm uses information from other phases or performs explicitly joint reasoning

Important interpretation rule used in this project:

- if a scheduler uses later-phase information to influence current scheduling decisions, it belongs to "our side"
- if it only solves each phase locally or ignores future-phase values, it belongs to the baseline side

Examples:

- baseline side:
  - `B_birkhoff`
  - `B_birkhoff_wave`
  - `B_barrier_aware_birkhoff`
  - `B_barrier_aware_birkhoff_wave`
  - `D_two_stage` was kept historically as a label, but semantically it is still a baseline-style comparator if it does not truly use future values
- our side:
  - `U_cp_lpt`
  - `U_lagrangian`
  - `U_ibbr`
  - `phase_aware_greedy`
  - `iterated_greedy`
  - `decomposed`
  - `cp_local_swap`
  - global matching family under `U_*`

Recent audit fix already applied:

- `phase_aware_greedy`, `decomposed`, `iterated_greedy`, `cp_local_swap`
  are now marked `prediction_aware=True` in `src/routesense/scheduler/strategies.py`

## 6. Offline Experiment Structure

### 6.1 Core question

Offline experiments operate on three traffic matrices:

- `M0 = dispatch`
- `M1 = combine`
- `M2 = next_dispatch`

Two semantic modes matter:

1. `execution_window`
   `M0`, `M1`, `M2` are all treated as real traffic in a 3-phase horizon.
2. `runtime_lookahead`
   `M2` is only predictive context and must not generate real current payload.

### 6.2 Dataset and prompt policy

The retained standard non-repeated corpus path is:

- `RS/artifacts/poc_line1/prompt_sets/olmoe_oasst256_unique.jsonl`

Tiered experiment config:

- `RS/experiments/poc_line1/configs/candidate_tiers.json`

### 6.3 Candidate tiers

The intended evaluation discipline is:

1. Tier 1:
   final contenders only, larger sample / larger N
2. Tier 2:
   promotion pool, medium sample / medium N
3. Tier 3:
   quick small-scale filter for new ideas

The user later tightened Tier 2 from sample `128` to sample `64`.

### 6.4 Important retained offline conclusions

The project intentionally retained only a few curated conclusion groups:

1. cross-layer prediction validity
2. oracle vs greedy / FAST optimization gap
3. execution-window multiscale scheduler study

The root `README.md` and `archive/backup/README.md` point to those curated backups.

## 7. Current Offline Status

What is already true:

- offline scheduler pipeline is implemented
- real OLMoE trace-derived experiments were run previously
- oracle / heuristic gap studies exist in backup artifacts
- candidate comparison infrastructure exists
- global matching U-family prototypes exist in the scheduler tree

What is not fully settled:

- full fairness alignment between chunk-level and wave-level execution semantics across all baselines
- final promotion / elimination table under the new tiering rule
- multi-model offline parity beyond OLMoE and partial Qwen work

## 8. Real Deployment Status

### 8.1 What works

The real execution mainline currently supports:

- model load
- real OLMoE trace collection
- full dispatch plan construction
- native baseline execution
- wave-collective execution path
- correctness checks against native baseline

Main entrypoint:

- `RS/experiments/distributed/exp_wave_execution.py`

Execution modes:

- `native_baseline`
- `wave_collective`

### 8.2 What was actually verified

A real single-node single-rank OLMoE run already succeeded on remote GPU with:

- `execution_mode=native_baseline`
- `compute_mode=actual_olmoe_expert`
- model path explicitly set to `/root/autodl-tmp/models/OLMoE-1B-7B-0924`

It also succeeded for:

- `execution_mode=wave_collective`
- `strategy=U_gated_maxweight_matching`

In the single-rank case there is no cross-GPU traffic, so wave count is `0` and correctness trivially matches the native path. This still matters because it proves the end-to-end injected execution path is live.

### 8.3 Important deployment bug fixes already done

Recent functional fixes:

1. `runner.py`
   full global `DispatchPlan` construction instead of rank-local only shard plans
2. `distributed_nccl_smoke.py`
   explicit asymmetric `all_to_all_single` smoke
3. `wave_planner.py`
   accepts both dict-style payloads and `SchedulingResult` dataclass objects
4. `scheduler/strategies.py`
   corrected `prediction_aware` metadata
5. `scheduler/local_search.py`
   removed duplicate `_clone_phase_orders` and `_best_insert_position`

### 8.4 Current blocker

True multi-node distributed execution is blocked by networking, not by the current RouteSense code path.

Observed fact:

- both remote environments expose only local docker-style addresses such as `172.17.0.x`
- direct TCP listener tests across nodes on those addresses fail with `ConnectionRefusedError`
- this strongly suggests the shown `172.17.*` addresses are per-host container bridge addresses, not a shared inter-node network

Implication:

- `torchrun --rdzv-endpoint=<172.17.x.x:port>` cannot currently be used for cross-machine rendezvous

What is needed:

1. true inter-node reachable IPs from the platform
2. or a platform-supported tunnel / port-forwarding workaround

## 9. Remote Testbed

Two remote nodes were prepared via `exp_link_smoke.py`.

Current known node access:

- node0:
  - SSH target: `root@connect.cqa1.seetacloud.com -p 34708`
  - GPU: `RTX 4090 D`
- node1:
  - SSH target: `root@connect.cqa1.seetacloud.com -p 21608`
  - GPU: `RTX 4090 D`

Remote repo path:

- `/root/autodl-tmp/RouterSense/RS`

Remote Python environment used successfully:

- `/root/miniconda3/bin/python`

Do not rely on:

- `/root/model_env/bin/python`

because it was missing the correct PyTorch / transformer stack during bring-up.

## 10. How To Run

### 10.1 Offline candidate comparison

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

### 10.2 Single-rank real OLMoE execution

```bash
cd /root/autodl-tmp/RouterSense/RS
PYTHONPATH=src /root/miniconda3/bin/torchrun --nnodes=1 --nproc_per_node=1 \
  experiments/distributed/exp_wave_execution.py \
  --execution-mode native_baseline \
  --compute-mode actual_olmoe_expert \
  --model allenai/OLMoE-1B-7B-0924 \
  --model-path /root/autodl-tmp/models/OLMoE-1B-7B-0924 \
  --layer-index 0
```

Wave-injected path:

```bash
cd /root/autodl-tmp/RouterSense/RS
PYTHONPATH=src /root/miniconda3/bin/torchrun --nnodes=1 --nproc_per_node=1 \
  experiments/distributed/exp_wave_execution.py \
  --execution-mode wave_collective \
  --strategy U_gated_maxweight_matching \
  --compute-mode actual_olmoe_expert \
  --model allenai/OLMoE-1B-7B-0924 \
  --model-path /root/autodl-tmp/models/OLMoE-1B-7B-0924 \
  --layer-index 0
```

### 10.3 NCCL smoke

```bash
cd /root/autodl-tmp/RouterSense/RS
PYTHONPATH=src /root/miniconda3/bin/torchrun --nnodes=1 --nproc_per_node=1 \
  experiments/distributed/exp_nccl_smoke.py
```

For true multi-node smoke, do not proceed until platform-provided reachable node IPs are confirmed.

## 11. Test Settings To Remember

Important recurring settings from the project discussion:

- offline main comparison often centered on:
  - `execution_window`
  - `N = 8, 16, 32`
  - sample limits like `32`, `64`, `200`, `500` depending on tier
- the project intentionally moved away from very large early runs before small-sample sanity checks
- wave vs atomic comparison matters
- oracle comparisons are only tractable on smaller cases
- `hidden_window_ms` and `token_to_ms_factor` were introduced for end-to-end net-benefit reporting

Current practical rule of thumb:

- use Tier 3 for quick rejection
- use Tier 2 for promotion checks
- use Tier 1 only for final contenders

## 12. Documentation To Read First

A next Codex should read these first:

1. `README.md`
2. `RS/docs/handoff_next_codex.md`
3. `RS/docs/multiphase_global_matching_study.md`
4. `RS/docs/phase0c_distributed_ep_contract.md`
5. `RS/docs/poc_line1_status.md`
6. `archive/backup/README.md`

Then inspect:

- `RS/src/routesense/scheduler/`
- `RS/src/routesense/runtime/distributed_ep/`
- `RS/experiments/distributed/`
- `RS/experiments/poc_line1/`

## 13. Known Risks And Open Questions

1. Multi-node rendezvous is blocked by missing inter-node reachable IPs.
2. OLMoE deployment path is ahead of Qwen/Mixtral/DeepSeek deployment support.
3. Some offline conclusions were produced under different fairness assumptions before wave-vs-atomic cleanup was fully settled.
4. The single-rank real execution path is proven; multi-rank real cross-node execution is not yet proven.
5. The project should avoid overstating production-scale hardware claims from 4090-class validation.

## 14. Suggested Next Steps

If the next Codex resumes this project, the best order is:

1. obtain real inter-node reachable addresses from platform support
2. rerun distributed NCCL smoke with those addresses
3. rerun `exp_wave_execution.py` in 2-node mode on OLMoE
4. only after distributed correctness is stable, expand to Qwen and Mixtral
5. keep DeepSeek-V2-Lite as the fourth-model target once weights and adapter path are ready

## 15. Minimal Handoff Summary

If you only remember five facts, remember these:

1. The mainline lives in `RS/`, not `legacy/`.
2. OLMoE single-rank real execution already works for both native and wave-injected paths.
3. Multi-node blocking issue is network addressability, not the current scheduler bridge.
4. The project now classifies algorithms by whether future-phase information actually influences current scheduling.
5. Use the curated archive and docs as the source of retained conclusions; ignore stale root-level historical task notes except for context.
