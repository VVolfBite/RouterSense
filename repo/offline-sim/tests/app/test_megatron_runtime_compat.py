from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import torch

from rs_sim.trace.collection.bootstrap import install_from_environment
from rs_sim.trace.runners.megatron_bridge_moe import (
    _BridgeWeightLoadAudit,
    _configure_provider,
    _forward_kwargs,
    _install_bridge_ep_none_task_compat,
)


def _args():
    return SimpleNamespace(
        tp=1,
        pp=1,
        ep=4,
        etp=1,
        cp=1,
        dtype="bfloat16",
        dispatcher="alltoall",
    )


def test_provider_configuration_preserves_family_expert_layout_defaults():
    class Provider:
        tensor_model_parallel_size = 1
        pipeline_model_parallel_size = 1
        expert_model_parallel_size = 1
        expert_tensor_parallel_size = 1
        context_parallel_size = 1
        params_dtype = torch.float32
        pipeline_dtype = torch.float32
        bf16 = False
        fp16 = False
        sequence_parallel = False
        moe_grouped_gemm = True
        moe_permute_fusion = True
        moe_token_dispatcher_type = "alltoall"
        cuda_graph_impl = "local"

    provider = Provider()
    configured = _configure_provider(provider, _args(), torch)
    assert provider.moe_grouped_gemm is True
    assert provider.moe_permute_fusion is True
    assert configured["moe_grouped_gemm"] is True
    assert configured["moe_permute_fusion"] is True
    assert provider.expert_model_parallel_size == 4


def test_forward_requests_gathered_logits_for_static_inference_context():
    class Model:
        def forward(self, tokens, position_ids=None, inference_context=None, runtime_gather_output=None):
            raise NotImplementedError

    kwargs = _forward_kwargs(
        Model(),
        {"tokens": object(), "position_ids": object(), "inference_context": object()},
    )
    assert kwargs["runtime_gather_output"] is True


def test_bridge_ep_none_task_compat_filters_rank_absent_tasks():
    live_task = object()

    class Bridge:
        def build_conversion_tasks(self, *args, **kwargs):
            return [None, live_task, None]

    bridge = Bridge()
    stats = _install_bridge_ep_none_task_compat(bridge)
    assert bridge.build_conversion_tasks("hf", "model") == [live_task]
    assert stats == {
        "installed": True,
        "total_task_count": 3,
        "non_null_task_count": 1,
        "null_task_count": 2,
    }


def test_weight_load_audit_fails_closed_on_unmapped_parameters():
    audit = _BridgeWeightLoadAudit()
    record = logging.LogRecord(
        name="megatron.bridge.models.conversion.model_bridge",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="No mapping found for megatron_param: decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight",
        args=(),
        exc_info=None,
    )
    audit.emit(record)
    report = audit.report(task_stats={"non_null_task_count": 10, "null_task_count": 2})
    assert report["status"] == "FAILED"
    assert report["critical_unmapped_parameter_count"] == 1
    assert report["mapped_conversion_task_count"] == 10


def test_torchrun_launcher_process_defers_capture_bootstrap(monkeypatch):
    monkeypatch.setenv("RS_SIM_CAPTURE_DEFER_TO_DISTRIBUTED_WORKERS", "1")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("RS_SIM_CAPTURE_CONFIG", raising=False)
    # Must return before importing Megatron or constructing a rank0/world1 session.
    assert install_from_environment() is None
