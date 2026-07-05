"""Artifact recording exports for the Megatron EP runtime."""

from __future__ import annotations

from rs.runtime.online.megatron_ep.trace_writer import write_json, write_jsonl

__all__ = ["write_json", "write_jsonl"]
