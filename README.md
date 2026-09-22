# When Does Fairness Calibration Transfer?
## Understanding Input-to-Output Exposure in Recommender Systems

This repository accompanies the ECIR 2027 submission. It provides (i) source code for the interventions and recommenders used in the study, (ii) **frozen machine-readable results** sufficient to regenerate paper tables and figures **without retraining**, and (iii) optional runners for full experiment reproduction.

---

## 1. Overview

We study whether an exposure-distribution change introduced **before** recommendation learning survives the learning and ranking pipeline. Interventions reweight training interactions toward a catalogue-share target \(T\); we measure how much of the induced input alignment appears in Top-\(K\) recommendation exposure.


## 2. Main experimental design

- **2** primary datasets: MovieLens-1M, Last.fm
- **5** recommenders: ItemKNN, BPR, NeuMF, LightGCN, NGCF
- **3** seeds: 0, 1, 2
- **8** interventions per baseline (4 inverse-power α + 4 KL \(\lambda_R\))
- **30** baselines + **240** matched intervention–baseline pairs
- **270** primary configurations total

Evidence tiers (never pool unlabeled): **PRIMARY**, **POST_LEARNING**, **TARGETED_EXTERNAL** (Gowalla–NGCF), **EXTERNAL_CASE_STUDY** (Amazon Books–LightGCN seed 0), **SUPPLEMENTARY**.

## 3. Repository structure

| Path | Role |
|---|---|
| `src/meg_rw/` | Grouping, \(T\), inverse-power, KL |
| `src/recbole_ext/` | Weighted RecBole backbones |
| `src/training/` | Training / MEG-RW injection |
| `src/evaluation/` | Full-sort evaluation, \(L_{\exp}\), Steck rerank |
| `src/analysis/` | Offline analysis helpers |
| `configs/` | Primary / external / post-learning grids |
| `results/` | Frozen CSVs/JSON by evidence tier |
| `figures/paper/` | Camera-ready PDFs (+ optional repro outputs) |
| `tables/paper/` | Frozen paper tables |
| `scripts/` | Verify / reproduce / optional full runs |
| `docs/` | Methods, datasets, reproducibility |
| `tests/` | Unit tests for core definitions |

## 4. Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
export PYTHONPATH=src

bash scripts/verify_artifact.sh
bash scripts/reproduce_tables.sh
bash scripts/reproduce_figures.sh
pytest -q
```

## 5. Full experiment reproduction

Training the full primary matrix (**270** configs, CPU) is **not** required to verify reported numbers.

See `docs/REPRODUCIBILITY.md` for:

- dataset preparation
- `scripts/run_primary_matrix.py` (primary grid)
- `scripts/run_steck_anchor.py` (post-learning; needs baseline checkpoints)
- external runners

## 6. Datasets

Raw datasets are **not** redistributed here. Obtain:

| Dataset | Variant used |
|---|---|
| MovieLens 1M | RecBole `ml-1m` |
| Last.fm | RecBole `lastfm` (HetRec2011 lineage) |
| Gowalla | LightGCN-PyTorch preprocessed dump → ECIR atomic files |
| Amazon Books | LightGCN-PyTorch preprocessed dump → ECIR atomic files |

Details: `docs/DATASETS.md`.

## 7. Intervention configurations (primary)

| Family | Parameters |
|---|---|
| Inverse-power | (alpha in {0.1, 0.2, 0.4, 0.8}) |
| Minimum-perturbation KL | (lambda_R in {0.1, 0.5, 2.0, 8.0}) |

Baseline: (alpha=0) / empty calibration mode (φ = 1).

## 8. Main results files

See `docs/RESULTS.md` for a claim→file map. Highlights:

- `results/primary/master_240.csv` — primary intervention pairs
- `results/primary/rq2_cells.csv` — dataset×model slopes
- `results/post_learning/matched_48.csv` — stage comparison
- `results/external/` — Gowalla / Amazon

## 9. Reproducibility notes

- Seeds `{0,1,2}`; RecBole `reproducibility: true`
- Split: user-grouped RS `[0.8, 0.1, 0.1]`, `order: RO`
- Full-sort evaluation; Top-(K=10)
