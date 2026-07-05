"""Formal online host-API probe entrypoint."""

from __future__ import annotations

from integrations.megatron_ep.probe_dispatch_boundary import main


if __name__ == "__main__":
    raise SystemExit(main())
