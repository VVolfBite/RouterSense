#!/usr/bin/env python3
from __future__ import annotations

"""Compatibility alias for the canonical deployment pipeline."""

from scripts.deploy.run_deployment_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
