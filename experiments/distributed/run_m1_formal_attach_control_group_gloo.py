from __future__ import annotations

import json
import os
import socket
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import rs.runtime.online.megatron_ep.host as host_mod
from rs.runtime.online.megatron_ep.contracts import ExecutionSelection, OnlinePolicyParameters, OnlineRuntimeConfig
from rs.runtime.online.megatron_ep.host import attach_formal_online_runtime
from rs.runtime.online.megatron_ep.public_types import FormalRuntimeAttachPreflightError


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Dispatcher:
    def __init__(self, ep_group, *, include_combine: bool = True) -> None:
        self.ep_group = ep_group
        self.token_dispatch = lambda *args, **kwargs: ("dispatch", args, kwargs)
        if include_combine:
            self.token_combine = lambda *args, **kwargs: ("combine", args, kwargs)


class _Layer(torch.nn.Module):
    def __init__(self, dispatcher) -> None:
        super().__init__()
        self.token_dispatcher = dispatcher


class _Container(torch.nn.Module):
    def __init__(self, dispatcher) -> None:
        super().__init__()
        self.mlp = _Layer(dispatcher)


class _Model(torch.nn.Module):
    def __init__(self, dispatcher) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Container(dispatcher), _Container(dispatcher)])

    def forward(self):
        self.layers[1].mlp.token_dispatcher.token_dispatch("hidden", "probs")
        self.layers[1].mlp.token_dispatcher.token_combine("hidden")
        return "ok"


def _config() -> OnlineRuntimeConfig:
    return OnlineRuntimeConfig(
        policy_name="routersense_p0p1p2_hint",
        execution_mode="native_passthrough",
        control_mode="sync_before_phase",
        execution_selection=ExecutionSelection(layer_selector="selected", selected_layer_ids=("1",)),
        policy_parameters=OnlinePolicyParameters(
            online_p2_predictor="copy_current_dispatch",
            safe_projection_mode="disabled",
        ),
    )


def _create_ep_groups(rank: int) -> tuple[tuple[int, ...], object]:
    ordered = ((0, 1), (2, 3))
    groups = {}
    for group_ranks in ordered:
        groups[group_ranks] = dist.new_group(ranks=list(group_ranks), backend="gloo")
    local_group_ranks = ordered[0] if rank in {0, 1} else ordered[1]
    return local_group_ranks, groups[local_group_ranks]


def _scenario_attach_success(rank: int) -> dict[str, object]:
    local_group_ranks, ep_group = _create_ep_groups(rank)
    dispatcher = _Dispatcher(ep_group, include_combine=True)
    model = _Model(dispatcher)
    handle = attach_formal_online_runtime(
        model=model,
        runtime_config=_config(),
        rank=rank,
        local_rank=rank,
        run_id=f"attach-success-{rank}",
        model_revision="model",
        request_table_hash="request",
        hostname="host",
    )
    try:
        registry = next(iter(host_mod._CONTROL_GROUP_REGISTRY.values()))
        return {
            "scenario": "attach_success",
            "rank": rank,
            "control_group_ranks": list(handle.runtime.target_plan_control_group_handle.group_ranks),
            "control_root_global_rank": int(handle.runtime.target_plan_control_group_handle.root_global_rank),
            "new_group_call_order": [list(item) for item in registry.new_group_call_order],
            "owner_present": getattr(model, "_routersense_runtime_owner", None) is not None,
        }
    finally:
        handle.close()
        assert getattr(model, "_routersense_runtime_owner", None) is None


def _scenario_attach_preflight_failure(rank: int) -> dict[str, object]:
    local_group_ranks, ep_group = _create_ep_groups(rank)
    dispatcher = _Dispatcher(ep_group, include_combine=(rank != 2))
    model = _Model(dispatcher)
    failure = None
    try:
        attach_formal_online_runtime(
            model=model,
            runtime_config=_config(),
            rank=rank,
            local_rank=rank,
            run_id=f"attach-fail-{rank}",
            model_revision="model",
            request_table_hash="request",
            hostname="host",
        )
    except FormalRuntimeAttachPreflightError as exc:
        failure = str(exc)
    return {
        "scenario": "attach_preflight_failure",
        "rank": rank,
        "failure": failure,
        "owner_present": getattr(model, "_routersense_runtime_owner", None) is not None,
        "registry_size": len(host_mod._CONTROL_GROUP_REGISTRY),
    }


def _validate(results: list[dict[str, object]]) -> None:
    success = [item for item in results if str(item.get("scenario")) == "attach_success"]
    failure = [item for item in results if str(item.get("scenario")) == "attach_preflight_failure"]
    assert len(success) == 4, results
    assert len(failure) == 4, results
    assert all(item["new_group_call_order"] == [[0, 1], [2, 3]] for item in success), success
    assert all(bool(item["owner_present"]) for item in success), success
    assert [item["control_root_global_rank"] for item in success if item["rank"] in {0, 1}] == [0, 0], success
    assert [item["control_root_global_rank"] for item in success if item["rank"] in {2, 3}] == [2, 2], success
    assert all(item["failure"] is not None for item in failure), failure
    assert all(item["owner_present"] is False for item in failure), failure
    assert all(int(item["registry_size"]) == 0 for item in failure), failure


def _worker(rank: int, world_size: int, master_port: int, out_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{int(master_port)}",
        rank=rank,
        world_size=world_size,
    )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    try:
        results = [
            _scenario_attach_success(rank),
            _scenario_attach_preflight_failure(rank),
        ]
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, {"rank": rank, "results": results})
        if rank == 0:
            flat = []
            for item in gathered:
                flat.extend(list((item or {}).get("results", [])))
            _validate(flat)
            summary = {"status": "passed", "world_size": world_size, "results": flat}
            (out_path / "m1_formal_attach_control_group_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary, indent=2))
        dist.barrier()
    except Exception as exc:
        failure = {
            "rank": rank,
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        (out_path / f"rank{rank}_m1_attach_failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise
    finally:
        dist.destroy_process_group()


def main() -> None:
    out_dir = "outputs/closure/m1_formal_attach_control_group_gloo"
    port = _free_port()
    mp.spawn(_worker, args=(4, port, out_dir), nprocs=4, join=True)


if __name__ == "__main__":
    main()
