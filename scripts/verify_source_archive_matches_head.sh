#!/usr/bin/env bash
set -euo pipefail

scope="mainline"
if [[ "${1:-}" == "--scope" ]]; then
  scope="${2:?missing scope value}"
  shift 2
fi
archive_path="${1:?missing archive path}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
expected_commit="$(git -C "$repo_root" rev-parse HEAD)"
archive_commit="$(tar -xOf "$archive_path" SOURCE_COMMIT.txt | tr -d '\n')"

if [[ "$archive_commit" != "$expected_commit" ]]; then
  echo "commit mismatch: archive=$archive_commit head=$expected_commit" >&2
  exit 1
fi

listing="$(tar -tzf "$archive_path")"
if [[ "$scope" == "mainline" ]]; then
  if grep -q '^legacy/poc1/' <<<"$listing"; then
    echo "mainline archive unexpectedly contains legacy/poc1" >&2
    exit 1
  fi
  if grep -q '^legacy/poc2/' <<<"$listing"; then
    echo "mainline archive unexpectedly contains legacy/poc2" >&2
    exit 1
  fi
fi

if grep -q 'hosts.local.yaml$' <<<"$listing"; then
  echo "archive unexpectedly contains local inventory" >&2
  exit 1
fi

echo "VERIFY_OK scope=$scope commit=$expected_commit"
