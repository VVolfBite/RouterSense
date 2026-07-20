# Current experiment inventory

Formal active entrypoints are intentionally small:

- `experiments/run_offline_replay.py`: typed offline replay;
- `scripts/verify/run_round1_offline_regression.py`: deterministic RSCF
  Local/Joint regression over EP4/8/12/16;
- `scripts/deploy/run_allready_pipeline.py`: deployment gate orchestration;
- `experiments/distributed/run_stage1_gloo_e2e_gate.py`: low-level distributed
  transport gate using a canonical deployable control;
- `scripts/diagnostics/replay_online_planner.py`: formal online planner replay;
- `scripts/diagnostics/analyze_prepared_plan_runtime.py`: prepared-plan timing
  and reconciliation analysis.

Historical experiment entrypoints are preserved only under
`archive/round1_removed_20260720` and are not importable from the installable
package.
