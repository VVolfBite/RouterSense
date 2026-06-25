from __future__ import annotations

import logging
from pathlib import Path

from .config import build_config, ensure_output_dir, parse_args
from .model_loader import load_olmoe_model
from .router_trace import build_batch_routing_summary, collect_router_trace
from .schemas import RunConfig
from .serialization import save_json
from .utils import ensure_directory


logger = logging.getLogger(__name__)


def _write_config_snapshot(config: RunConfig, output_dir: Path) -> None:
    save_json(config.to_dict(), output_dir / "config.json")


def _write_goal(output_dir: Path, title: str, body: str) -> None:
    (output_dir / "goal.md").write_text(f"# {title}\n\n{body}\n", encoding="utf-8")


def run_inspect_model(config: RunConfig) -> int:
    output_dir = ensure_output_dir(config.output_dir)
    _write_config_snapshot(config, output_dir)
    ensure_directory("logs")
    logger.info("真实 OLMoE trace 主线已接入 run_action.py trace / trace_one_token.py。")
    logger.info("inspect 入口当前仍未实现真实模型结构导出。")
    save_json(
        {
            "status": "not_implemented",
            "completed_mainline": ["real_router_trace"],
            "message": "真实 router trace 已完成主线接入；inspect 仍是未完成入口，当前不会假装导出真实模型结构。",
        },
        output_dir / "environment.json",
    )
    save_json(
        {
            "status": "not_implemented",
            "completed_mainline": ["real_router_trace"],
            "message": "inspect 未完成。请使用 scripts/run_action.py trace 或 scripts/trace_one_token.py 生成真实 trace 输出。",
        },
        output_dir / "model_inspection.json",
    )
    save_json(
        {
            "status": "not_implemented",
            "message": "adapter/inspect 仍未完成；当前完成的是真实 OLMoE router trace 导出链路。",
        },
        output_dir / "adapter_status.json",
    )
    return 0


def run_trace_one_token(config: RunConfig) -> int:
    output_dir = ensure_output_dir(config.output_dir)
    _write_config_snapshot(config, output_dir)
    _write_goal(
        output_dir,
        "POC1 Trace Single",
        "Validate real OLMoE loading and export one real router trace plus a minimal batch routing summary.",
    )
    ensure_directory("logs")
    if config.mock:
        trace = collect_router_trace(model=None, tokenizer=None, text=config.text, layer_path=config.layer, use_mock=True)
        mock_trace_payload = trace.to_dict()
        mock_trace_payload["status"] = "mock"
        mock_trace_payload["is_mock"] = True
        save_json(mock_trace_payload, output_dir / "mock_one_sample_trace.json")
        save_json(trace.to_dict(), output_dir / "one_sample_trace.json")
        save_json(build_batch_routing_summary(trace).to_dict(), output_dir / "batch_routing_summary.json")
        return 0
    model, tokenizer = load_olmoe_model(config)
    trace = collect_router_trace(model=model, tokenizer=tokenizer, text=config.text, layer_path=config.layer, use_mock=False)
    save_json(trace.to_dict(), output_dir / "one_sample_trace.json")
    save_json(build_batch_routing_summary(trace).to_dict(), output_dir / "batch_routing_summary.json")
    return 0


def run_ablate_one_route(config: RunConfig) -> int:
    output_dir = ensure_output_dir(config.output_dir)
    _write_config_snapshot(config, output_dir)
    ensure_directory("logs")
    logger.info("真实 OLMoE trace 主线已完成；真实 route ablation 尚未实现。")
    save_json(
        {
            "status": "not_implemented",
            "completed_mainline": ["real_router_trace"],
            "message": "真实 route ablation 尚未实现。当前项目状态是：trace 已真实跑通，ablate 仍明确未完成。",
        },
        output_dir / "ablation_not_run.json",
    )
    return 0


def main(argv: list[str] | None = None, command: str | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    if command == "trace":
        return run_trace_one_token(config)
    if command == "ablate":
        return run_ablate_one_route(config)
    return run_inspect_model(config)


if __name__ == "__main__":
    raise SystemExit(main())
