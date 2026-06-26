# RouteSense POC1

RouteSense POC1 is a MoE inference systems prototype. The current mainline is **lossless dependency-aware scheduling**: keep the model output semantics and Top-K routing unchanged, but use router-visible batch structure before communication to identify likely short-board buckets or destinations.

This repository is now past the local trace bring-up stage. The local evidence is still proxy-level evidence, not final runtime proof.

## Current Status

Completed:

- Real OLMoE router trace export.
- Real `batch_routing_summary`.
- Multi-sample `trace_batch_summary`.
- Proxy comparison v1/v2/v3/v4.
- Critical bucket proxy v1/v2.
- Prompt category grouping.
- Proxy result verification.
- Windows smoke test.

Explicitly not completed:

- Real single-route ablation.
- Adapter / route patch integration.
- Full runtime scheduler.
- Server-scale experiments.

The current local conclusion is:

- Dependency-aware and full proxies remain worth pursuing.
- `critical_bucket_proxy_v2` keeps the pseudo ground truth more load-centric than v1, and still shows dependency/full ahead of state-only.
- On the latest local 48-prompt x 3-layer batch, top-1 hit rates under v2 are roughly: state `0.9236`, dependency `0.9722`, full `0.9861`.
- `layer_15` is currently the most useful layer to prioritize for the next decision-level experiments, while `layer_0` is not a good place to spend much effort.
- These are proxy results. They support continuing the mainline, but do not replace runtime validation.

## Important Entrypoints

Current experiment entrypoints live under `experiment/poc1/`. Older `scripts/` entrypoints are kept as compatibility wrappers, platform smoke scripts, or implementation helpers reused by the thin experiment wrappers.

Directory responsibilities:

- `src/routesense_poc1/`: core library logic for config, model loading, trace extraction, schemas, and serialization.
- `experiment/poc1/`: main POC1 experiment entrypoints.
- `deploy/`: deployment-facing wrappers and new-machine run notes.
- `scripts/`: legacy compatibility layer and platform utilities; do not add new main experiments here.
- `docs/`: result templates, metric notes, and project documentation.
- `tests/`: regression tests for the current code contracts.

Single real trace:

```bash
python experiment/poc1/trace_single.py --text "The history of science is a story of" --layer auto --model-id model/OLMoE
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

## Linux Default Flow

```bash
bash deploy/run_poc1_linux.sh
```

The Linux script keeps the flow conservative:

1. setup environment
2. preflight
3. inspect status
4. real trace

It does not run ablation by default because real route ablation is not implemented yet.

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

## POC2 Runtime Audit

The current POC2 emphasis is a runtime leverage audit for the NCCL harness rather than a larger policy sweep.
The audit asks whether policy decisions materially change release-round composition, NCCL split sizes, bottleneck ranks, and end-to-end completion under the current synchronous collective structure.
This stage is about explaining `full ~= random-order` and locating the bottleneck source, not about claiming dependency-aware success.
