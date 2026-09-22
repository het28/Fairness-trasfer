# Reproducibility

## Software requirements

Declared in `requirements.txt` / `pyproject.toml`:

- Python ≥ 3.9
- PyTorch ≥ 2.0
- RecBole ≥ 1.2.0
- NumPy, SciPy, Pandas, PyYAML, Matplotlib, pytest

**Observed during artifact build** (informative, not a guarantee): torch 2.4.x, RecBole 1.2.1.

## Seeds and determinism

- Primary seeds: `{0,1,2}`
- RecBole `reproducibility: true`
- Bootstrap for gain-space slope CI: `n_boot=2000`, seed `42`
- No stronger claim of bit-identical GPU training (primary matrix is **CPU**)

## FAST REPRODUCTION (recommended for reviewers)

No training. Uses frozen CSVs under `results/`.

```bash
pip install -r requirements.txt
pip install -e .
export PYTHONPATH=src
bash scripts/verify_artifact.sh
bash scripts/reproduce_tables.sh
bash scripts/reproduce_figures.sh
pytest -q
```

Expected:

- verifier exits 0
- `tables/paper/*.csv` refreshed
- `figures/paper/*_repro.pdf` written (camera-ready `fig_*.pdf` already included)

## FULL REPRODUCTION (optional, expensive)

1. Obtain datasets (`docs/DATASETS.md`); set `data_path`.
2. Primary matrix: adapt `scripts/run_primary_matrix.py` paths to this repo layout (`configs/primary/`, local `dataset/`, output under `runs/`), then run. Grid: `configs/primary/experiment_grid.yaml`.
3. Post-learning: requires baseline checkpoints for LightGCN/NGCF; `configs/post_learning/steck_grid.yaml`; runner `scripts/run_steck_anchor.py` (path adaptation required).
4. External: `configs/external/` + `scripts/run_gowalla_external.py` / `scripts/run_amazon_external.py`.

**Reviewers do not need full reproduction to validate reported numbers.**

## Grids (authoritative)

| Scope | Grid |
|---|---|
| Primary interventions | α∈{0.1,0.2,0.4,0.8}; λ_R∈{0.1,0.5,2.0,8.0} |
| Primary size | 2×5×3×(1+8)=270 |
| Post-learning λ | {0,0.1,0.25,0.5,0.75,0.9,1.0} → 84 runs → 48 matched |
| Gowalla | NGCF; seeds 0–2; α=0.8 + λ_R=0.1 |
| Amazon | LightGCN seed 0; α∈{0.2,0.8}, λ_R∈{0.1,2.0} |

## Package naming note

Training/evaluation packages retain historical module names `cikm_train` / `cikm_eval` to avoid import-breaking renames. Scientific behavior is unchanged. See `docs/ARTIFACT_MANIFEST.md`.
