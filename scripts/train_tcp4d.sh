#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec conda run --no-capture-output -n dp3 python "$REPO_DIR/scripts/train_tcp4d.py" "$@"
