#!/usr/bin/env bash
# Fast path: verify frozen numbers (no training).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
python scripts/verify_artifact.py
