# Results map (paper claims → frozen files)

Evidence scopes are labeled. Do not pool scopes.

## PRIMARY

| Analysis / claim | File |
|---|---|
| 240 intervention pairs | `results/primary/master_240.csv` |
| Full 270 configs (incl. baselines) | `results/primary/transfer_metrics_270.csv` |
| Baselines summary | `results/primary/baselines.csv` |
| RQ1 global (ρ, τ, slope, CI, means) | `results/primary/rq1_global.csv` |
| Intervention families | `results/primary/rq1_by_family.csv` |
| Dataset×model cells (Table 1 style) | `results/primary/rq2_cells.csv` |
| Architecture / dataset aggregates | `results/primary/rq2_architectures.csv`, `rq2_datasets.csv` |
| M0/M1/M2 R² | `results/primary/rq2_regression.csv` |
| Attenuation / reversal | `results/primary/rq3_failures.csv` |
| Utility trade-off | `results/primary/utility_tradeoff.csv` |
| Contribution / frozen R² JSON | `results/primary/contribution_validation_summary.json` |
| Fig. realization data | `results/primary/figure_data/fig_realization.csv` |
| Fig. cell slopes data | `results/primary/figure_data/fig_cell_slopes.csv` |
| Fig. stage data | `results/primary/figure_data/fig_stage.csv` |

## POST_LEARNING

| Analysis | File |
|---|---|
| 84 Steck runs | `results/post_learning/steck_84.csv` |
| 48 matched comparisons | `results/post_learning/matched_48.csv` |
| Aggregates | `results/post_learning/matched_agg.csv` |

## TARGETED_EXTERNAL / EXTERNAL_CASE_STUDY

| Analysis | File |
|---|---|
| Gowalla–NGCF | `results/external/gowalla_ngcf.csv` |
| Amazon Books–LightGCN | `results/external/amazon_books.csv` |
| Combined export table | `results/external/external_summary.csv` |

## USER_LEVEL

| Analysis | File |
|---|---|
| Headline answers (99.5%, ρ, p90/p95) | `results/user_level/answers.json` |
| Per-run summaries | `results/user_level/summary.csv` |
| By user group | `results/user_level/by_usergroup.csv` |

## Figures (camera-ready)

| Figure | File |
|---|---|
| Realization scatter | `figures/paper/fig_realization.pdf` |
| Cell slopes | `figures/paper/fig_cell_slopes.pdf` |
| Stage comparison | `figures/paper/fig_stage.pdf` |

Regenerate from CSVs: `bash scripts/reproduce_figures.sh` → `*_repro.pdf`.

## Headline number check

`bash scripts/verify_artifact.sh` validates the frozen numerical claims listed in the build instructions (n=240, means, correlations, R², utility, post means, external means, user-level rates).
