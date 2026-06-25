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

- `outputs/poc1_trace_single/`
- `outputs/poc1_trace_batch/`
- `outputs/poc1_proxy_compare/`

Large outputs stay local and are ignored by git.
