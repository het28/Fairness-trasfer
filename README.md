# When Does Fairness Calibration Transfer?
## Understanding Input-to-Output Exposure in Recommender Systems

This repository accompanies the ECIR 2027 submission. It provides (i) source code for the interventions and recommenders used in the study, (ii) **frozen machine-readable results** sufficient to regenerate paper tables and figures **without retraining**, and (iii) optional runners for full experiment reproduction.

---

## 1. Overview

We study whether an exposure-distribution change introduced **before** recommendation learning survives the learning and ranking pipeline. Interventions reweight training interactions toward a catalogue-share target \(T\); we measure how much of the induced input alignment appears in Top-\(K\) recommendation exposure.

## 2. Core quantities

| Symbol | Meaning |
|---|---|
| \(q_{\mathrm{in}}\) | Training interaction-mass distribution over popularity groups |
| \(E_{\mathrm{out}}\) | Aggregate Top-\(K\) exposure share over the same groups |
| \(F_{\mathrm{in}}, F_{\mathrm{out}}\) | \(\tfrac12\|\,\cdot\,-T\|_1\) (total variation / L1 deviation from \(T\)) |
| InputGain | \(F_{\mathrm{in}}^{\mathrm{base}}-F_{\mathrm{in}}\) |
| OutputGain | \(F_{\mathrm{out}}^{\mathrm{base}}-F_{\mathrm{out}}\) |
| \(\Gamma\) | OutputGain / InputGain (descriptive realization ratio) |

**Target \(T\)** is the **empirical catalogue-share** of four popularity groups. It is a controlled reference used for measurement—not a claim that catalogue share is a universal normative definition of fairness.

## 3. Main experimental design

- **2** primary datasets: MovieLens-1M, Last.fm
- **5** recommenders: ItemKNN, BPR, NeuMF, LightGCN, NGCF
- **3** seeds: 0, 1, 2
- **8** interventions per baseline (4 inverse-power α + 4 KL \(\lambda_R\))
- **30** baselines + **240** matched intervention–baseline pairs
- **270** primary configurations total

Evidence tiers (never pool unlabeled): **PRIMARY**, **POST_LEARNING**, **TARGETED_EXTERNAL** (Gowalla–NGCF), **EXTERNAL_CASE_STUDY** (Amazon Books–LightGCN seed 0), **SUPPLEMENTARY**.

## 4. Repository structure

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

## 5. Quick start

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

## 6. Reproducing the paper WITHOUT retraining

Frozen results under `results/` support all reported primary, post-learning, external, and user-level numbers.

```bash
bash scripts/verify_artifact.sh      # exits nonzero on mismatch
bash scripts/reproduce_tables.sh     # refreshes tables/paper/
bash scripts/reproduce_figures.sh    # writes figures/paper/*_repro.pdf
```

Camera-ready figures shipped with the artifact:

- `figures/paper/fig_realization.pdf`
- `figures/paper/fig_cell_slopes.pdf`
- `figures/paper/fig_stage.pdf`

## 7. Full experiment reproduction

Training the full primary matrix (**270** configs, CPU) is **not** required to verify reported numbers.

See `docs/REPRODUCIBILITY.md` for:

- dataset preparation
- `scripts/run_primary_matrix.py` (primary grid)
- `scripts/run_steck_anchor.py` (post-learning; needs baseline checkpoints)
- external runners

## 8. Datasets

Raw datasets are **not** redistributed here. Obtain:

| Dataset | Variant used |
|---|---|
| MovieLens 1M | RecBole `ml-1m` |
| Last.fm | RecBole `lastfm` (HetRec2011 lineage) |
| Gowalla | LightGCN-PyTorch preprocessed dump → ECIR atomic files |
| Amazon Books | LightGCN-PyTorch preprocessed dump → ECIR atomic files |

Details: `docs/DATASETS.md`.

## 9. Intervention configurations (primary)

| Family | Parameters |
|---|---|
| Inverse-power | \(\alpha \in \{0.1, 0.2, 0.4, 0.8\}\) |
| Minimum-perturbation KL | \(\lambda_R \in \{0.1, 0.5, 2.0, 8.0\}\) |

Baseline: \(\alpha=0\) / empty calibration mode (φ = 1).

## 10. Models and weight consumption

| Model | How \(w=\phi(g(i))\,c_{ui}\) enters |
|---|---|
| ItemKNN | Interaction matrix values for similarity |
| BPR / NeuMF | Per-sample **loss** weights |
| LightGCN / NGCF | Weighted bipartite adjacency |

## 11. Main results files

See `docs/RESULTS.md` for a claim→file map. Highlights:

- `results/primary/master_240.csv` — primary intervention pairs
- `results/primary/rq2_cells.csv` — dataset×model slopes
- `results/post_learning/matched_48.csv` — stage comparison
- `results/external/` — Gowalla / Amazon

## 12. Supplementary results

Additional verified tables (reversal case list, full user-level table, utility strength curves, full Steck λ trajectories) are under `results/supplementary/` and `results/user_level/`. See `docs/SUPPLEMENTARY_RESULTS.md`.

## 13. Reproducibility notes

- Seeds `{0,1,2}`; RecBole `reproducibility: true`
- Split: user-grouped RS `[0.8, 0.1, 0.1]`, `order: RO`
- Full-sort evaluation; Top-\(K=10\)
- Bootstrap for gain slope: \(n=2000\), seed `42`
- Primary matrix: **CPU** (`use_gpu: false`)
- ML-1M: effective \(c_{ui}=1\); LastFM: \(c_{ui}=\log(1+w_{ui})\)

## 14. Expected runtime

Training runtimes are **not** claimed here (no authoritative wall-clock summary is packaged). Fast verification (tables/figures from frozen CSVs) completes in seconds to a few minutes on a laptop.

## 15. Artifact scope and limitations

- External experiments are **not** full factorial.
- Amazon Books is LightGCN **seed 0** only (case study).
- Catalogue-share \(T\) is a controlled target.
- Post-learning is an **adapted** Steck-style catalogue calibrator (shared \(T\)), not an exact reproduction of Steck (2018) user-preference calibration.
- Row-level bootstrap does **not** cluster observations that share a baseline.

## 16. Citation

Citation information will be added after the review process.
