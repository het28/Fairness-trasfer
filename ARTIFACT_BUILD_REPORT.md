# ARTIFACT_BUILD_REPORT

Anonymous ECIR 2027 research artifact build report.

**Local destination:** sibling directory `Fairness-trasfer/` (outside the source research repository).  
**GitHub:** not pushed; no remote configured; README uses `<ANONYMOUS_REPOSITORY_URL>`.  
**Source repository:** not modified by this build.

---

## 1. Final directory tree (top level)

```
README.md
LICENSE_NOTES.md
requirements.txt
pyproject.toml
.gitignore
configs/{primary,external,post_learning}/
src/{meg_rw,recbole_ext,training,evaluation,analysis}/
scripts/
results/{primary,post_learning,external,user_level,supplementary}/
figures/{paper,supplementary}/
tables/paper/
docs/
tests/
data/
ARTIFACT_BUILD_REPORT.md
```

## 2. Files copied from source

- `src/meg_rw/*`, `src/recbole_ext/*`, `src/training/*`, `src/evaluation/*`
- Primary/external YAML configs (not resolved absolute-path dumps)
- Frozen paper export CSVs/JSON (primary, post, external, user-level, supplementary)
- Camera-ready figure PDFs/PNGs
- Unit tests under `tests/`
- Optional full-repro runners (path-adapted)
- `requirements.txt` dependency pins (ranges)

## 3. Files newly created

- `README.md` (anonymous)
- `docs/*` (inventory, methods, datasets, reproducibility, results maps, anonymity audit, manifest)
- `configs/*/experiment_grid.yaml`, `steck_grid.yaml`
- `scripts/verify_artifact.{py,sh}`, `reproduce_{tables,figures}.{py,sh}`, wrappers
- `LICENSE_NOTES.md`, `.gitignore`, `pyproject.toml` (anonymous package name)
- `ARTIFACT_BUILD_REPORT.md`

## 4. Deliberately excluded

- Source `.git` history
- `clean_matrix/` (~12GB checkpoints / eval dumps)
- Steck `cand_*.npz` score caches (~1.3GB)
- Raw datasets
- Prior-venue legacy experiment dumps
- Stale metadata claiming raw-rating \(c_{ui}\), leave-one-out, wrong KL, 11-λ Steck grid
- IDE files, wandb, env secrets
- Author-identifying CITATION.cff

## 5. Stale docs excluded

- `dataset_preprocessing.md` (leave-one-out / wrong filters / wrong LastFM c_ui)
- `intervention_implementation.md` (wrong KL direction / scipy / sampling claim)
- Unchecked “ML-1M raw rating” sentences from export prose (facts corrected in `docs/METHODS.md`)

## 6. Tests executed

```text
export PYTHONPATH=src
pytest -q
→ 22 passed
```

## 7. Paper-number verification

```text
python scripts/verify_artifact.py
→ PASS (all listed PRIMARY / POST / EXTERNAL / USER checks)
```

## 8. Figure reproduction

```text
python scripts/reproduce_figures.py
→ wrote figures/paper/*_repro.pdf (+ PNG)
```

Camera-ready `fig_realization.pdf`, `fig_cell_slopes.pdf`, `fig_stage.pdf` included.  
No programmatic pipeline schematic found in exports; paper conceptual Fig.1 (if any) is not claimed as data-generated here.

## 9. Table reproduction

```text
python scripts/reproduce_tables.py
→ refreshed tables/paper/*.csv from frozen results/
```

## 10. Anonymity audit

See `docs/ANONYMITY_AUDIT.md`. Absolute local paths removed from runners and Steck checkpoint column. PDF metadata = Matplotlib only.

## 11. Remaining manual actions

1. Push this local tree to the **anonymous** review remote when ready (do not use a personal profile URL in docs).
2. Confirm dataset license text with venue if needed.
3. After acceptance: add LICENSE + CITATION.cff with authors.
4. Optional: provide anonymized baseline checkpoints if reviewers request full Steck re-run (not required for number checks).
5. Full-repro runners still need a local `dataset/` layout; document paths already point to `configs/` and `runs/`.

## 12. Repository size

Approximately **2.7 MB** (excluding `.git`).

## 13. Large files

None above a few hundred KB except figure PDFs/PNGs and `transfer_metrics_270.csv` / user-level tables. No checkpoints.

## 14. Dataset licensing caveats

Raw MovieLens / Last.fm / Gowalla / Amazon Books dumps are **not** redistributed. External experiments use LightGCN-preprocessed dumps (documented in `docs/DATASETS.md`).

## 15. Unresolved issues

- Package directories remain `training` / `evaluation` for import stability (historical names; not personal IDs).
- Optional analysis aggregators are not the primary reviewer path; prefer `verify_artifact.sh`.
- Gowalla optional runner source file historically targeted seeds `{1,2}` as a continuation job; **frozen** `results/external/gowalla_ngcf.csv` remains the authoritative 3-seed evidence.

---

ARTIFACT_STATUS = READY_FOR_ANONYMOUS_GITHUB
