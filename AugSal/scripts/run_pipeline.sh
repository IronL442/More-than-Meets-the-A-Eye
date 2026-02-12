#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-AugSal/configs/default.yaml}"
shift || true

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" AugSal/pipeline.py --config "${CONFIG_PATH}" "$@"
