#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

module="${1:?missing module, e.g. experiments.online.run_policy_correctness}"
shift || true

exec python -m "${module}" "$@"
