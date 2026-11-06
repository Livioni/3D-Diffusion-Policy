#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_DIR/3D-Diffusion-Policy${PYTHONPATH:+:$PYTHONPATH}"
exec conda run --no-capture-output -n dp3 python -m diffusion_policy_3d.deploy.server "$@"
