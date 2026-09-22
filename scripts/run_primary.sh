#!/usr/bin/env bash
# Optional full primary matrix (expensive; CPU). Not required to verify paper numbers.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
echo "Full primary training is expensive (270 configs)."
echo "Prefer: bash scripts/verify_artifact.sh"
echo "To launch training, see docs/REPRODUCIBILITY.md (FULL REPRODUCTION)."
echo "Runner entry point: python scripts/run_primary_matrix.py"
echo "(Paths inside that script may need local dataset/ layout; see docs/DATASETS.md.)"
