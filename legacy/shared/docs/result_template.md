# Result Directory Template

Each experiment should write results under:

```text
outputs/<experiment_name>/
```

Recommended minimal contents:

1. `goal.md`
   - One to three short sentences.
   - Say what the experiment tries to validate and what this batch of results roughly represents.

2. `summary.json`
   - The main aggregate result for quick review.
   - If the experiment already has a more specific summary file, `summary.json` may be a small pointer file that names it.

3. `metric_notes.md`
   - Short explanations of the metrics used by this experiment.
   - It should help future readers understand what to inspect first.

4. Sample-level outputs, if needed
   - Put sample traces or per-sample summaries in a clearly named subdirectory such as `trace_batch/`.

Current POC1 examples:

- `outputs/poc1_trace_single/`: one real OLMoE trace and its minimal batch routing summary.
- `outputs/poc1_trace_batch/`: multi-prompt, multi-layer real trace summaries and per-sample trace files.
- `outputs/poc1_proxy_compare/`: state/dependency/full proxy comparison, verification, and critical bucket proxy summaries.

For current POC1 work, do not write new main results directly into `outputs/`. Create or reuse an experiment subdirectory and put the main summary there.

If a directory has both `summary.json` and a more specific file such as `trace_batch_summary.json` or `critical_bucket_proxy_summary.json`, treat `summary.json` as the short pointer and the specific file as the full result.

Large outputs stay local and are ignored by git.
