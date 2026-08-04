# RouterSense Final Workspace

This workspace collects the final paper package, formal experiment run kit,
formal-ready simulator source, online implementation source, and complete trace
bundle.

## Top-level layout

- `article/`: final compacted INFOCOM manuscript package.
- `run_kit/`: formal experiment runbook, scripts, matrix, prompts, and validation assets.
- `repo/`: formal-ready RS-SIM simulator source and source archive.
- `online_impl/`: v152 P12 online implementation source and source archive.
- `trace/`: complete measured/projected trace archive and extracted trace tree.

## Main artifacts

| Area | File | Purpose |
|---|---|---|
| article/archive | `ReleaseFrontier_INFOCOM_FORMULATION_COMPACTED_20260803.zip` | Final compacted manuscript/source package |
| article/source | extracted `ReleaseFrontier_INFOCOM_FORMULATION_COMPACTED_20260803/` tree | Active paper workspace |
| run_kit | `RouterSense_FORMAL_EXPERIMENT_RUN_KIT_20260801.zip` plus extracted files | Formal experiment instructions and helper scripts |
| repo/dist | `RouterSense_RS_SIM_FORMAL_READY_SOURCE_20260801.zip` | Formal-ready RS-SIM source package |
| repo/source | extracted formal-ready RS-SIM source tree | Active simulator workspace |
| online_impl/archive | `RouterSense_v152_p12_source_ready_milestone_c_full_source_20260727.zip` | v152 P12 online implementation source archive |
| online_impl/source | extracted `RouterSense_v152_p12_source_ready_milestone_c_full_source/` tree | Active online implementation workspace |
| trace/archive | `RouterSense_COMPLETE_TRACE_EP2_EP4_EP8_EP16_EP32_20260801.tar.gz` | Complete measured EP2/EP4/EP8 plus projected EP16/EP32 trace bundle |
| trace/data | extracted trace tree | Active trace workspace |

## Evidence boundary

- Measured traces cover EP2/EP4/EP8.
- EP16/EP32 traces are projected evidence and must be labelled `PROJECTED_NOT_MEASURED`.
- The run kit is not the simulator source; it is kept separately under `run_kit/`.
- Checksums for the current main archives are recorded in `CHECKSUMS.sha256`.
