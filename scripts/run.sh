#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
python -m saliency_bench.core.runner --config "$1"

