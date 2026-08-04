# One-command durable formal sweep

The recommended entry point needs only a trace repository and an output CSV:

```bash
python scripts/run_formal_experiment.py \
  --trace-root /path/to/RouterSense_COMPLETE_TRACE_EP2_EP4_EP8_EP16_EP32_20260801 \
  --output-csv /path/to/results/formal_results.csv \
  --preset main \
  --trace-kind measured \
  --workers 2
```

Installed form:

```bash
rs-sim-formal \
  --trace-root /path/to/trace_repository \
  --output-csv /path/to/results/formal_results.csv
```

Presets:

- `smoke`: FIFO-Local, Birkhoff-Local, RSCF-Joint-FATE;
- `main`: FIFO, Greedy, Birkhoff, iSLIP, Residual-MWM/GMWD-style, FAST, Aurora, and RSCF Local/Zero/FATE/Perfect;
- `oracle`: FIFO/Birkhoff/RSCF comparators plus Local/Joint Oracle references;
- `paper`: `main` plus both Oracle references in one run;
- `all`: all Local and Joint genericity variants.

The default formal task size is 256 KiB. Use `--task-bytes 4194304` for a fast
4 MiB full-trace smoke before the formal run.

Trace selection:

- `--trace-kind measured` is the default and excludes projected EP16/EP32 traces;
- `--trace-kind projected` runs only projected scaling traces;
- `--trace-kind all` combines both evidence classes in one CSV.

When `--runtime-profile` is omitted, the simulator uses the built-in
`SYNTHETIC_TEST_ONLY` profile and keeps `claim_mode=DIAGNOSTIC`. This must not
be reported as hardware-calibrated performance.

## Durable output and recovery

The output CSV is an append-only journal. Every completed treatment row is
written by the parent process, followed by `flush()` and `fsync()`. Every fully
processed fixture emits a `TRACE_COMPLETE` row. The sidecars
`<csv>.progress.json` and `<csv>.manifest.json` are atomically replaced and
fsynced.

The default `--resume` mode reconstructs committed run keys from the CSV and
skips completed PASS rows. Re-running exactly the same command after a crash or
manual interruption continues from the first incomplete treatment. Use
`--no-resume` only when a fresh CSV is required.

Multiple isolated workers may execute concurrently, but only the parent writes
the CSV, so rows are never interleaved or partially written.

## Validation behavior

Each fixture receives full schema, invariant, and truth-digest validation once
in the parent process. Isolated workers receive a trusted descriptor containing
size, modification time, and truth digest; they verify that the file has not
changed and avoid repeating the full invariant walk for every treatment. The
worker exports `observation__fixture_validation_mode` and validation elapsed
time for audit.

## Trace discovery

A directory, manifest, ZIP, TAR, TAR.GZ, or TGZ repository is accepted. Discovery
is recursive and loads fixtures referenced by `trace_manifest.json`, or JSON
files below `fixtures/` when manifests are absent. Multiple `--trace-root`
arguments are supported.

## Plotting

```bash
pip install -e '.[plot]'
python scripts/plot_results.py \
  --input-csv /path/to/results/formal_results.csv \
  --output-dir /path/to/figures \
  --baseline FIFO-Local
```

The plotting entry point exports normalized CSVs, per-window and per-rank data,
PDF/SVG/PNG figures, a compact main table, and a LaTeX table.
