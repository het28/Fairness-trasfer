#!/usr/bin/env bash
# Analyze from frozen transfer metrics (no training).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
python scripts/verify_artifact.py
python scripts/reproduce_tables.py
echo "Primary analysis verification complete (frozen results)."
