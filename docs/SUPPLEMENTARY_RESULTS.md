# Supplementary results

These artifacts are **verified exports** supporting the paper but not all shown in a 12-page main text. Scopes remain labeled.

## Included supplementary material

| Item | Path | Scope | Why useful |
|---|---|---|---|
| Full reversal case list | `results/supplementary/reversal_cases.csv` | PRIMARY | All 29 operational reversals with context |
| Regime counts | `results/supplementary/regime_counts.csv` | PRIMARY | Taxonomy counts |
| Utility vs strength figure data | `results/supplementary/fig_utility_data.csv` | PRIMARY | InputGain vs relative NDCG loss |
| Utility figure | `figures/supplementary/fig_utility.pdf` | PRIMARY | Visual companion |
| Full user-level table | `results/user_level/table_user_level.csv` | USER_LEVEL | Per configuration L_exp summaries |
| Full Steck λ trajectories | `results/post_learning/steck_84.csv` | POST_LEARNING | Beyond the 48 matched rows |
| Intervention strength table | `results/primary/intervention_strength.csv` | PRIMARY | Per-α / per-λ_R aggregates |
| Cell CI export | `results/supplementary/cell_slopes_ci.csv` | PRIMARY | Alternate CI table if present |

## Explicitly not included

- Abandoned prior-venue experiment dumps
- Full `clean_matrix` checkpoints / eval dumps (~12GB)
- Steck candidate score caches (`cand_*.npz`)
- Speculative / incomplete pilot folders

## How to cite supplementary evidence in review

Always name the evidence tier (PRIMARY / POST_LEARNING / TARGETED_EXTERNAL / EXTERNAL_CASE_STUDY / SUPPLEMENTARY) when discussing numbers.
