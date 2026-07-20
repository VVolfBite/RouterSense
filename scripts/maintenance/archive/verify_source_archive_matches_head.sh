#!/usr/bin/env bash
set -euo pipefail

scope="mainline"
if [[ "${1:-}" == "--scope" ]]; then
  scope="${2:?missing scope value}"
  shift 2
fi
archive_path="${1:?missing archive path}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
expected_commit="$(git -C "$repo_root" rev-parse HEAD)"
manifest_json="$(tar -xOf "$archive_path" source_manifest.json)"
archive_commit="$(python -c 'import json,sys; print(json.load(sys.stdin)["commit_sha"])' <<<"$manifest_json")"

if [[ "$archive_commit" != "$expected_commit" ]]; then
  echo "commit mismatch: archive=$archive_commit head=$expected_commit" >&2
  exit 1
fi

listing="$(tar -tzf "$archive_path")"
if [[ "$scope" == "mainline" ]] && grep -q '^RS/legacy/' <<<"$listing"; then
  echo "mainline archive unexpectedly contains RS/legacy" >&2
  exit 1
fi

if grep -q '\\' <<<"$listing"; then
  echo "archive contains non-POSIX path separators" >&2
  exit 1
fi

echo "VERIFY_OK scope=$scope commit=$expected_commit"
