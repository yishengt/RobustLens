#!/usr/bin/env bash
# Run the original RobustLens checkpoint on the local-edit test set.
# Usage: bash scripts/run_raw_test.sh       # first 20 images
#        bash scripts/run_raw_test.sh 0     # all test images

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python"
fi

LIMIT="${1:-20}"
TIMESTAMP="$(date -u +"%Y%m%d_%H%M%S")"
ARGS=(
  scripts/run_inference.py
  --input-dir "$ROOT/data/local_edits/test"
  --checkpoint "$ROOT/models/pretrained/pytorch_model.pt"
  --config "$ROOT/configs/config.yaml"
  --device mps
  --no-transformations
  --no-patches
  --no-calibration
  --output "$ROOT/outputs/raw_test_${TIMESTAMP}.json"
  --detailed-output "$ROOT/outputs/raw_test_detailed_${TIMESTAMP}.json"
)

if [[ "$LIMIT" != "0" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

cd "$ROOT"
exec "$PYTHON" "${ARGS[@]}"
