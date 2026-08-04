from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


class ModelSupportError(ValueError):
    """Raised before GPU launch when a checkpoint cannot use the generic runner."""


_HIDDEN_SIZE_KEYS = ("hidden_size", "d_model", "n_embd", "model_dim")
_VOCAB_SIZE_KEYS = ("vocab_size", "padded_vocab_size")
_LAYER_COUNT_KEYS = ("num_hidden_layers", "num_layers", "n_layer", "decoder_layers")
_EXPERT_COUNT_KEYS = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "num_routed_experts",
    "moe_num_experts",
)
_TOPK_KEYS = ("num_experts_per_tok", "moe_router_topk", "top_k", "router_topk")

# Decoder-only text MoE families whose Megatron implementations use the common
# MCore MoELayer lifecycle.  The installed AutoBridge remains authoritative:
# this list is used for diagnostics, not to bypass AutoBridge capability checks.
_KNOWN_TEXT_MOE_MARKERS = (
    "olmoe",
    "mixtral",
    "deepseek",
    "qwen2moe",
    "qwen3moe",
    "qwen3next",
    "gptoss",
    "gpt_oss",
    "glm",
    "minimax",
    "moonlight",
    "kimi",
    "bailing",
    "ling",
    "sarvam",
    "step",
    "nemotron",
    "mimo",
)
_MULTIMODAL_MARKERS = (
    "vision",
    "vl",
    "audio",
    "omni",
    "multimodal",
    "image",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelSupportError(f"Hugging Face config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelSupportError(f"invalid Hugging Face config JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelSupportError(f"model config must be a JSON object: {path}")
    return value


def _config_candidates(root: dict[str, Any]) -> Iterable[dict[str, Any]]:
    # Text sub-configs come first for VLM-style configs.  Multimodal models are
    # still rejected by the generic runner, but this gives useful diagnostics.
    for key in ("text_config", "language_config", "llm_config", "decoder_config"):
        value = root.get(key)
        if isinstance(value, dict):
            yield value
    yield root


def _first_int(configs: Iterable[dict[str, Any]], keys: Iterable[str]) -> int | None:
    for config in configs:
        for key in keys:
            value = config.get(key)
            if isinstance(value, bool) or value is None:
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                return parsed
    return None


def _architectures(root: dict[str, Any]) -> tuple[str, ...]:
    value = root.get("architectures")
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(item) for item in value if str(item).strip())
    return ()


def _model_type(root: dict[str, Any]) -> str:
    for config in _config_candidates(root):
        value = config.get("model_type")
        if value:
            return str(value)
    return ""


def _detect_expert_count(root: dict[str, Any]) -> int | None:
    configs = tuple(_config_candidates(root))
    # Prefer explicit total routed-expert fields.  num_experts_per_tok is only
    # top-k and must not be mistaken for the total when another field exists.
    explicit = _first_int(
        configs,
        (
            "num_experts",
            "num_local_experts",
            "n_routed_experts",
            "num_routed_experts",
            "moe_num_experts",
        ),
    )
    if explicit is not None:
        return explicit
    # Some configs nest MoE settings under ffn_config/moe_config.
    nested: list[dict[str, Any]] = []
    for config in configs:
        for key in ("moe_config", "ffn_config", "expert_config"):
            value = config.get(key)
            if isinstance(value, dict):
                nested.append(value)
    return _first_int(nested, _EXPERT_COUNT_KEYS)


def _detect_moe_layer_count(root: dict[str, Any], total_layers: int | None) -> int | None:
    configs = tuple(_config_candidates(root))
    for config in configs:
        pattern = config.get("moe_layer_freq")
        if isinstance(pattern, list):
            return sum(1 for item in pattern if bool(item))
        if isinstance(pattern, int) and pattern > 0 and total_layers:
            first_dense = int(config.get("first_k_dense_replace", 0) or 0)
            return sum(
                1
                for index in range(total_layers)
                if index >= first_dense and (index - first_dense) % pattern == 0
            )
    if total_layers is None:
        return None
    first_dense = 0
    for config in configs:
        if config.get("first_k_dense_replace") is not None:
            try:
                first_dense = max(0, int(config["first_k_dense_replace"]))
            except (TypeError, ValueError):
                pass
            break
    return max(0, total_layers - first_dense)


@dataclass(frozen=True)
class ModelInspection:
    model_path: str
    config_path: str
    model_type: str
    architectures: tuple[str, ...]
    hidden_size: int | None
    vocab_size: int | None
    total_layers: int | None
    estimated_moe_layers: int | None
    num_experts: int | None
    router_topk: int | None
    is_probably_moe: bool
    is_decoder_only_text: bool
    runner_kind: str = "MEGATRON_BRIDGE_AUTO_TEXT_MOE"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["architectures"] = list(self.architectures)
        return value


def inspect_hf_model(model_path: str | Path) -> ModelInspection:
    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise ModelSupportError(f"model.path is not a directory: {root}")
    config_path = root / "config.json"
    config = _read_json(config_path)
    candidates = tuple(_config_candidates(config))
    architectures = _architectures(config)
    model_type = _model_type(config)
    hidden_size = _first_int(candidates, _HIDDEN_SIZE_KEYS)
    vocab_size = _first_int(candidates, _VOCAB_SIZE_KEYS)
    total_layers = _first_int(candidates, _LAYER_COUNT_KEYS)
    num_experts = _detect_expert_count(config)
    router_topk = _first_int(candidates, _TOPK_KEYS)
    estimated_moe_layers = _detect_moe_layer_count(config, total_layers)

    identity = " ".join((*architectures, model_type)).lower().replace("-", "").replace("_", "")
    probable_moe = bool(
        (num_experts is not None and num_experts > 1)
        or "moe" in identity
        or any(marker.replace("_", "") in identity for marker in _KNOWN_TEXT_MOE_MARKERS)
    )
    multimodal = any(marker in identity for marker in _MULTIMODAL_MARKERS)
    causal = any("causallm" in item.lower() for item in architectures) or not architectures
    decoder_text = causal and not multimodal

    return ModelInspection(
        model_path=str(root),
        config_path=str(config_path),
        model_type=model_type,
        architectures=architectures,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        total_layers=total_layers,
        estimated_moe_layers=estimated_moe_layers,
        num_experts=num_experts,
        router_topk=router_topk,
        is_probably_moe=probable_moe,
        is_decoder_only_text=decoder_text,
    )


def validate_generic_text_moe(
    inspection: ModelInspection,
    *,
    ep: int,
    require_moe: bool = True,
) -> None:
    if not inspection.is_decoder_only_text:
        raise ModelSupportError(
            "the built-in runner supports decoder-only text MoE checkpoints; "
            f"detected architectures={list(inspection.architectures)} model_type={inspection.model_type!r}. "
            "Vision/audio/omni models require modality-specific inputs and are rejected before GPU launch."
        )
    if require_moe and not inspection.is_probably_moe:
        raise ModelSupportError(
            "model config does not identify a MoE architecture; refusing to collect a dense model as MoE"
        )
    if inspection.num_experts is not None and inspection.num_experts % int(ep) != 0:
        raise ModelSupportError(
            f"num_experts={inspection.num_experts} is not divisible by EP={int(ep)}"
        )
    if inspection.estimated_moe_layers is not None and inspection.estimated_moe_layers < 2:
        raise ModelSupportError(
            f"Current-P12 requires at least two MoE layers; config estimates {inspection.estimated_moe_layers}"
        )
