# RouteSense POC1

RouteSense is a MoE systems prototype. POC1/POC2 remain frozen as research artifacts; the active deployment line is **RS Phase 0B**, which brings up real single-GPU OLMoE inference and router-trace capture on the executor host, while the control host manages inventory, launch contracts, and log collection.

The phase 0B work is intentionally conservative: it does not claim multi-node expert parallel, does not change router output, and does not add a scheduler benchmark.

## Current Status

Completed historical artifacts:

- Real OLMoE router trace export.
- Real `batch_routing_summary`.
- Multi-sample `trace_batch_summary`.
- Proxy comparison v1/v2/v3/v4.
- Critical bucket proxy v1/v2.
- Prompt category grouping.
- Proxy result verification.
- Windows smoke test.

Active phase 0B work:

- Two-host inventory templates.
- Dry-run repo/model sync scripts.
- Single-GPU OLMoE text inference smoke.
- Single-GPU real router-trace smoke.
- Future 2-node x 2-GPU launch contract dry-run.

Explicitly not completed in phase 0B:

- Multi-node expert parallel.
- Cross-node NCCL dispatch.
- Scheduler performance benchmark.
- Adapter patching or ablation.


## Important Entrypoints

Current experiment entrypoints live under `experiment/poc1/` and `experiments/deployment/`. Older `scripts/` entrypoints are kept as compatibility wrappers, platform smoke scripts, or implementation helpers reused by the thin experiment wrappers.

Directory responsibilities:

- `src/routesense_poc1/`: frozen POC1 core library logic for config, model loading, trace extraction, schemas, and serialization.
- `src/routesense/`: formal deployment-side package for phase 0B runtime, trace, topology, and artifact helpers.
- `experiment/poc1/`: main POC1 experiment entrypoints.
- `experiments/deployment/`: phase 0B deployment smoke and trace entrypoints.
- `deploy/`: control-plane inventory, dry-run scripts, and remote launch contracts.
- `scripts/`: legacy compatibility layer and platform utilities; do not add new main experiments here.
- `docs/`: result templates, metric notes, and project documentation.
- `tests/`: regression tests for the current code contracts.

Single real trace (POC1 historical):

```bash
python experiment/poc1/trace_single.py --text "The history of science is a story of" --layer auto --model-id model/OLMoE
```

Phase 0B single-GPU OLMoE smoke:

```bash
python experiments/deployment/single_gpu_olmoe_smoke.py
```

Phase 0B real router trace smoke:

```bash
python experiments/deployment/single_gpu_router_trace_smoke.py
```

Compatibility action entry:

```bash
python scripts/run_action.py trace --text "The history of science is a story of" --layer auto --model-id model/OLMoE
```

Windows smoke test:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\run_poc1_windows.ps1 -UseCurrentPython -ModelId model\OLMoE -OutputDir outputs\poc1_windows_smoke
```

Trace batch:

```bash
python experiment/poc1/trace_batch.py --model-id model/OLMoE --max-samples 48 --layers 0,8,15
```

Proxy comparison:

```bash
python experiment/poc1/proxy_compare.py --top-k 3
```

Critical bucket proxy:

```bash
python experiment/poc1/critical_bucket_proxy.py
```

Proxy result verification:

```bash
python experiment/poc1/verify_proxy_results.py
```

Server-prep trace entry, for a future larger run:

```bash
python experiment/poc1/server_trace_batch.py --model-id model/OLMoE
python experiment/poc1/proxy_compare.py --input-dir outputs/poc1_server_trace_batch --output-dir outputs/poc1_server_proxy_compare --top-k 3
```

The server-prep entry is still trace/proxy only. It does not run ablation or a real scheduler.

## Output Layout

Experiment outputs use:

```text
outputs/<experiment_name>/
```

Main local directories:

- `outputs/poc1_trace_single/`
- `outputs/poc1_trace_batch/`
- `outputs/poc1_proxy_compare/`

Each experiment directory should contain:

- `goal.md`: one to three short sentences about the experiment.
- `summary.json`: the primary aggregate result or a short pointer summary.
- `metric_notes.md`: local notes for the metrics in that experiment.
- Optional sample-level subdirectories, when needed.

See `docs/result_template.md` for the expected result directory template and `docs/metric_notes.md` for metric definitions.

`outputs/` is intentionally ignored by git because real traces and summaries can be large and machine-local.

## Recommended Files To Inspect

For the current local phase, start with:

- `outputs/poc1_trace_batch/trace_batch_summary.json`
- `outputs/poc1_proxy_compare/proxy_comparison_summary.json`
- `outputs/poc1_proxy_compare/critical_bucket_proxy_summary.json`
- `outputs/poc1_proxy_compare/proxy_result_verification.json`

## Phase 0B Deployment Flow

```bash
bash deploy/scripts/check_cluster_access.sh
bash deploy/scripts/check_repo_parity.sh
bash deploy/scripts/launch_remote.sh
```

The phase 0B flow is deliberately conservative:

1. validate control-plane reachability
2. verify repository parity
3. dry-run the future 2-node launch contract
4. run single-GPU OLMoE smoke and router-trace smoke on the executor host when GPU access is available

## Compatibility Scripts

These `scripts/` files are intentionally kept, but they are not the preferred place for new experiment entrypoints:

- `scripts/trace_one_token.py`: legacy wrapper for single real trace.
- `scripts/run_action.py`: legacy action dispatcher.
- `scripts/summarize_trace_batch.py`: implementation helper used by `experiment/poc1/trace_batch.py` and server trace wrappers.
- `scripts/analyze_proxy_scores.py`: implementation helper used by `experiment/poc1/proxy_compare.py`.
- `scripts/run_poc1_linux.sh` and `scripts/run_poc1_windows.ps1`: platform scripts called by `deploy/` wrappers.

New POC1 experiment commands should go under `experiment/poc1/`.

## Repository Hygiene

The repository ignores model weights, Hugging Face caches, local outputs, logs, virtual environments, and local archives. Keep `model/OLMoE`, `outputs/`, `hf_cache/`, and `.venv/` local.

## Common Failure Points

- Model path is wrong or weights are incomplete.
- CUDA PyTorch is not installed in the active environment.
- Local GPU/CPU memory is insufficient for OLMoE loading.
- The installed Transformers version cannot expose OLMoE router logits through `output_router_logits=True`.
## Source Packaging

Use `scripts/package_source_only.sh` to create a delivery archive. The source package intentionally excludes `outputs/`, `artifacts/`, logs, caches, model files, and benchmark results. Runtime results stay on the server working directory or in separate result archives; they should not be mixed into the source tarball.

The current source package includes the formal `src/routesense/`, `deploy/`, `configs/`, `experiments/deployment/`, `docs/`, and `tests/` pieces needed for phase 0B.

## POC2 Runtime Audit

The current POC2 emphasis is a runtime leverage audit for the NCCL harness rather than a larger policy sweep.
The audit asks whether policy decisions materially change release-round composition, NCCL split sizes, bottleneck ranks, and end-to-end completion under the current synchronous collective structure.
This stage is about explaining `full ~= random-order` and locating the bottleneck source, not about claiming dependency-aware success.
