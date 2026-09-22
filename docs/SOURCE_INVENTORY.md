# Source Inventory (Phase 1)

Anonymous ECIR 2027 artifact. Source repository inspected read-only; nothing in the source was modified by this build.

Authoritative methodology reference: source `research_ecir2027/audit/FINAL_METHODS_REPRODUCIBILITY_AUDIT.md`.

Legend: **INCLUDED** = copied into this artifact. Relative DESTINATION paths are under this repository root.

---

## A–F. Dataset prep, groups, T, c_ui, interventions

| SOURCE PATH | DESTINATION PATH | PURPOSE | AUTHORITATIVE? | INCLUDED? | REASON |
|---|---|---|---|---|---|
| `src/meg_rw/grouping.py` | `src/meg_rw/grouping.py` | Popularity groups | yes | yes | Core |
| `src/meg_rw/statistics.py` | `src/meg_rw/statistics.py` | Catalog share T, dominance, mass | yes | yes | Core |
| `src/meg_rw/calibration.py` | `src/meg_rw/calibration.py` | Inverse-power + KL | yes | yes | Core |
| `src/meg_rw/reweight.py` | `src/meg_rw/reweight.py` | Legacy phi helpers | yes | yes | Core |
| `src/meg_rw/semantic.py` | `src/meg_rw/semantic.py` | Semantic path (disabled in ECIR) | yes | yes | Completeness; default off |
| `src/meg_rw/__init__.py` | `src/meg_rw/__init__.py` | Package | yes | yes | |
| `src/cikm_train/recbole_dataset.py` | `src/cikm_train/recbole_dataset.py` | c_ui + inject φ | yes | yes | Core |
| `research_ecir2027/experiments/prepare_amazon_gowalla_ecir.py` | `scripts/prepare_external_datasets.py` | External dataset prep | yes | yes | Docs + optional full repro |
| `research_ecir2027/paper/final_exports/metadata/dataset_preprocessing.md` | — | Stale (LOO, wrong c_ui) | no | **no** | Conflicts with audit |
| `research_ecir2027/paper/final_exports/metadata/intervention_implementation.md` | — | Stale KL / sampling | no | **no** | Conflicts with audit |

## G–K. Models

| SOURCE PATH | DESTINATION PATH | PURPOSE | AUTHORITATIVE? | INCLUDED? | REASON |
|---|---|---|---|---|---|
| `src/recbole_ext/weighted_itemknn.py` | `src/recbole_ext/weighted_itemknn.py` | ItemKNN | yes | yes | |
| `src/recbole_ext/weighted_bpr.py` | `src/recbole_ext/weighted_bpr.py` | BPR | yes | yes | |
| `src/recbole_ext/weighted_neumf.py` | `src/recbole_ext/weighted_neumf.py` | NeuMF | yes | yes | |
| `src/recbole_ext/weighted_lightgcn.py` | `src/recbole_ext/weighted_lightgcn.py` | LightGCN | yes | yes | |
| `src/recbole_ext/weighted_ngcf.py` | `src/recbole_ext/weighted_ngcf.py` | NGCF | yes | yes | |
| `src/recbole_ext/adj_utils.py` | `src/recbole_ext/adj_utils.py` | Weighted adj | yes | yes | |
| `src/recbole_ext/__init__.py` | `src/recbole_ext/__init__.py` | Package | yes | yes | |

## L–P. Training, eval, metrics, user-level

| SOURCE PATH | DESTINATION PATH | PURPOSE | AUTHORITATIVE? | INCLUDED? | REASON |
|---|---|---|---|---|---|
| `src/cikm_train/run_experiment.py` | `src/cikm_train/run_experiment.py` | Training entry | yes | yes | |
| `src/cikm_train/experiment_artifacts.py` | `src/cikm_train/experiment_artifacts.py` | Artifacts | yes | yes | |
| `src/cikm_train/__init__.py` | `src/cikm_train/__init__.py` | Package | yes | yes | |
| `src/cikm_eval/*.py` | `src/cikm_eval/*.py` | Full-sort audit, L_exp, Steck | yes | yes | All modules |
| `research_ecir2027/experiments/analyze_fairness_transfer.py` | `src/analysis/analyze_fairness_transfer.py` | F_in/F_out aggregation | yes | yes | Analysis |

## Q–T. Stats, Steck, external

| SOURCE PATH | DESTINATION PATH | PURPOSE | AUTHORITATIVE? | INCLUDED? | REASON |
|---|---|---|---|---|---|
| `research_ecir2027/paper/results_export/_build_paper_package.py` | `src/analysis/build_paper_tables.py` | Tables from frozen CSV | yes | yes | Fast repro |
| `research_ecir2027/experiments/run_steck_postprocessing_anchor.py` | `scripts/run_steck_anchor.py` | Post-learning | yes | yes | Full repro optional |
| `research_ecir2027/experiments/run_clean_matrix.py` | `scripts/run_primary_matrix.py` | Primary 270 grid | yes | yes | Full repro optional |
| `research_ecir2027/experiments/run_gowalla_ngcf_replication.py` | `scripts/run_gowalla_external.py` | Gowalla | yes | yes | |
| `research_ecir2027/experiments/run_external_transfer.py` | `scripts/run_amazon_external.py` | Amazon | yes | yes | |
| `research_ecir2027/results/.../postprocessing_steck/cand_*.npz` | — | Score caches | no | **no** | 1.3G; not needed for frozen numbers |
| `research_ecir2027/results/.../clean_matrix/` | — | Full train artifacts | yes (runs) | **no** | ~12G checkpoints/eval; frozen CSVs suffice |

## U–V. Tables and figures

| SOURCE PATH | DESTINATION PATH | PURPOSE | AUTHORITATIVE? | INCLUDED? | REASON |
|---|---|---|---|---|---|
| `paper/results_export/TABLE_*.csv` | `tables/paper/` + `results/primary/` | Paper tables | yes | yes | |
| `paper/results_export/MASTER_PRIMARY_RESULTS.csv` | `results/primary/master_240.csv` | 240 pairs | yes | yes | |
| `paper/results_export/figures/FIG{1,2,4}_*.csv` | `results/primary/figure_data/` | Figure inputs | yes | yes | |
| `paper/results_export/figures_final/*_FINAL.pdf` | `figures/paper/` | Final LNCS figures | yes | yes | PDF+PNG |
| `paper/results_export/figures_final/FIG_UTILITY_*` | `figures/supplementary/` | Utility figure | yes | yes | Supp |
| Pipeline schematic (if any) | — | Conceptual Fig.1 | ? | document if absent | No programmatic pipeline fig in exports |

## Frozen results (PRIMARY / POST / EXTERNAL / USER)

| SOURCE PATH | DESTINATION PATH | PURPOSE | AUTHORITATIVE? | INCLUDED? | REASON |
|---|---|---|---|---|---|
| `.../fairness_transfer/transfer_metrics.csv` | `results/primary/transfer_metrics_270.csv` | Full 270 | yes | yes | |
| `.../contribution_validation_summary.json` | `results/primary/` | M0–M2 R² | yes | yes | |
| `.../per_user_exposure_*.{csv,json}` | `results/user_level/` | L_exp | yes | yes | |
| `.../postprocessing_steck/steck_anchor_results.csv` | `results/post_learning/steck_84.csv` | 84 runs | yes | yes | |
| `.../stage_comparison_matched_utility.csv` | `results/post_learning/matched_48.csv` | 48 matches | yes | yes | |
| `.../gowalla_ngcf_seeds012.csv` | `results/external/gowalla_ngcf.csv` | External | yes | yes | |
| `paper/results_export/external_amazon_books.csv` | `results/external/amazon_books.csv` | Case study | yes | yes | |
| `paper/results_export/rq3_reversal_cases.csv` | `results/supplementary/reversal_cases.csv` | All reversals | yes | yes | Supp |
| `paper/results_export/TABLE_USER_LEVEL.csv` | `results/user_level/` | Full user-level table | yes | yes | |

## Configs

| SOURCE PATH | DESTINATION PATH | PURPOSE | AUTHORITATIVE? | INCLUDED? | REASON |
|---|---|---|---|---|---|
| `config/cikm_base.yaml` | `configs/primary/base.yaml` | Shared eval/train | yes | yes | Sanitize comments |
| `config/cikm_ml1m_*.yaml` | `configs/primary/ml1m_*.yaml` | Primary models | yes | yes | |
| `config/cikm_lastfm_*.yaml` | `configs/primary/lastfm_*.yaml` | Primary models | yes | yes | |
| `config/cikm_gowalla_ecir_ngcf.yaml` | `configs/external/gowalla_ngcf.yaml` | External | yes | yes | |
| `config/cikm_amazonbooks_ecir_lightgcn.yaml` | `configs/external/amazon_books_lightgcn.yaml` | External | yes | yes | |
| `clean_matrix/**/config_resolved.yaml` | — | Resolved with absolute paths | yes | **no** | Identity leak; recreate clean configs |

## Tests & methodology docs

| SOURCE PATH | DESTINATION PATH | PURPOSE | AUTHORITATIVE? | INCLUDED? | REASON |
|---|---|---|---|---|---|
| `tests/test_calibration.py` | `tests/test_calibration.py` | KL / inverse-power | yes | yes | |
| `tests/test_c_ui_and_l_exp.py` | `tests/test_c_ui_and_l_exp.py` | c_ui + L_exp | yes | yes | |
| `tests/test_meg_rw.py` | `tests/test_meg_rw.py` | Groups / phi | yes | yes | |
| `tests/test_adj_utils.py` | `tests/test_adj_utils.py` | Adj | yes | yes | |
| `tests/test_transfer_delta.py` | `tests/test_transfer_delta.py` | Gains | yes | yes | |
| Other tests | `tests/` | Supporting | yes | yes | If self-contained |
| `research_ecir2027/audit/FINAL_METHODS_REPRODUCIBILITY_AUDIT.md` | `docs/METHODS_AUDIT_SUMMARY.md` | Methods truth | yes | yes | Anonymized excerpt/facts only |
| `research_ecir2027/reports/METHOD_FREEZE_AND_CLEAN_PIPELINE.md` | — | Internal freeze | yes | **partial** | Facts folded into METHODS.md; skip CIKM path names identifying prior venue work where possible |

## Deliberately excluded

| SOURCE PATH | REASON |
|---|---|
| Entire `legacy prior-venue experiment dumps/` | Abandoned / prior-venue legacy |
| `clean_matrix/` (~12G) | Checkpoints + eval dumps; not needed for table/figure verification |
| `cand_*.npz` Steck caches | Large; not needed for frozen CSVs |
| Stale metadata (`dataset_preprocessing.md`, wrong KL docs) | Conflicts with FINAL audit |
| `PAPER_RESULTS_AUTHORITATIVE.md` unchecked c_ui claim | Stale ML-1M c_ui; numbers folded via CSVs |
| Source `.git`, IDE files, wandb | Anonymity / secrets |
| Raw `dataset/` dumps | License / redistribution; document how to obtain |
| Source README (CIKM / author-facing) | Identifying / stale |

## Conflicts vs FINAL_METHODS_REPRODUCIBILITY_AUDIT.md

No new conflicts discovered during inventory beyond those already documented in the audit (stale docs excluded). Copying proceeds with audit-correct methodology only.
