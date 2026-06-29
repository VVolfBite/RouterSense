from __future__ import annotations

import json

from .analysis.analysis import analyze_records, write_report
from .analysis.calibration import evaluate_calibrator, train_calibrator
from .core.schemas import AblationRecord
from .core.serialization import save_json
from .experiment.ablation import run_ablation
from .experiment.config import build_config, ensure_output_dir, finalize_config, parse_args
from .experiment.data import load_and_prepare_data
from .experiment.environment import run_doctor
from .runtime.model_loader import load_olmoe_model
from .runtime.moe_inspect import discover_moe_layers
from .runtime.trace import collect_routing_context


def _collect_trace_rows(config) -> list[dict]:
    model, tokenizer = load_olmoe_model(config)
    windows = load_and_prepare_data(config, tokenizer)
    moe_layers = discover_moe_layers(model, config.num_moe_layers)
    rows: list[dict] = []
    for window in windows:
        for moe_layer in moe_layers:
            rows.extend(
                context.to_dict()
                for context in collect_routing_context(
                    model=model,
                    tokenizer=tokenizer,
                    window=window,
                    moe_layer=moe_layer,
                    run_id=config.run_id,
                )
            )
    return rows


def run_command(command: str, config) -> int:
    output_dir = ensure_output_dir(config.output_dir)
    save_json(config.to_dict(), output_dir / "config.json")

    if command == "doctor":
        save_json(run_doctor(config), output_dir / "environment.json")
        return 0

    if command == "prepare-data":
        model, tokenizer = load_olmoe_model(config)
        windows = load_and_prepare_data(config, tokenizer)
        save_json([window.to_dict() for window in windows], output_dir / "selected_windows.json")
        return 0

    if command == "inspect-model":
        model, _ = load_olmoe_model(config)
        moe_layers = discover_moe_layers(model, config.num_moe_layers)
        save_json([layer.to_dict() for layer in moe_layers], output_dir / "moe_layers.json")
        return 0

    if command == "collect-trace":
        save_json(_collect_trace_rows(config), output_dir / "router_trace.json")
        return 0

    if command == "ablate":
        model, tokenizer = load_olmoe_model(config)
        windows = load_and_prepare_data(config, tokenizer)
        moe_layers = discover_moe_layers(model, config.num_moe_layers)
        records = run_ablation(
            config=config,
            model=model,
            tokenizer=tokenizer,
            windows=windows,
            moe_layers=moe_layers,
            output_dir=output_dir,
            resume=config.resume,
        )
        save_json([record.to_dict() for record in records], output_dir / "ablation_results.json")
        return 0

    if command == "calibrate":
        payload = json.loads((output_dir / "ablation_results.json").read_text(encoding="utf-8"))
        records = [AblationRecord(**row) for row in payload]
        calibrator = train_calibrator(records, config, output_dir)
        metrics = evaluate_calibrator(calibrator, records, output_dir=output_dir)
        save_json(metrics, output_dir / "calibration_metrics.json")
        return 0

    if command == "analyze":
        calibrator = None
        payload = json.loads((output_dir / "ablation_results.json").read_text(encoding="utf-8"))
        records = [AblationRecord(**row) for row in payload]
        calibrator_path = output_dir / "calibrator.joblib"
        if calibrator_path.exists():
            import joblib

            calibrator = joblib.load(calibrator_path)
        summary = analyze_records(records, calibrator=calibrator)
        save_json(summary, output_dir / "summary.json")
        write_report(summary, output_dir / "report.md")
        return 0

    if command == "run-all":
        run_command("doctor", config)
        run_command("prepare-data", config)
        run_command("inspect-model", config)
        run_command("collect-trace", config)
        run_command("ablate", config)
        if config.enable_calibration:
            run_command("calibrate", config)
        run_command("analyze", config)
        return 0

    raise ValueError(f"unsupported command: {command}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = finalize_config(build_config(args))
    return run_command(args.command, config)


if __name__ == "__main__":
    raise SystemExit(main())
