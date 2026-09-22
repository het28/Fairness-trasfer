# Artifact manifest

## Package layout decisions

| Decision | Rationale |
|---|---|
| Packages: `training`, `evaluation` | Clear package names |
| Do not ship `clean_matrix/` | Size; frozen CSVs suffice for reported numbers |
| Sanitize `steck_84.csv` checkpoint column | Removed absolute local paths; basename under `checkpoints/` |
| Exclude stale metadata docs | Wrong c_ui / split / KL / Steck grid |
| New scripts: `verify_artifact`, `reproduce_*` | Fast reviewer path without inventing scientific results |

## Import path

```bash
export PYTHONPATH=src
# or: pip install -e .
```

## Config derivation

Primary / external / post-learning grids were derived from the authoritative runners and resolved-run parameters (α / λ_R / seeds / Steck λ), not invented.

## Figure regeneration

Camera-ready PDFs are included. `scripts/reproduce_figures.py` regenerates LNCS-style plots from frozen figure CSVs. Visual styling may differ slightly from camera-ready PDFs; **data** are identical to the figure CSVs.
