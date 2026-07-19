#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT=${1:?experiment yaml required}
DEPLOYMENT=${2:?deployment yaml required}
MODE=${3:-dry-run}
case "$MODE" in
  dry-run) EXECUTE=() ;;
  execute|--execute) EXECUTE=(--execute) ;;
  *) echo "third argument must be dry-run or execute" >&2; exit 2 ;;
esac
routersense run --experiment "$EXPERIMENT" --deployment "$DEPLOYMENT" --mode deploy-all "${EXECUTE[@]}"
