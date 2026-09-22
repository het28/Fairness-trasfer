# Anonymity audit

Performed after constructing the anonymous artifact. Matches are inspected; this report does **not** reprint private identifying values.

| CHECK | RESULT | FILES INSPECTED | ACTION |
|---|---|---|---|
| Absolute user-home paths | PASS (after fix) | All text files; found in optional full-repro runner | Replaced with `Path(__file__).resolve().parents[1]`; checkpoint regex generalized |
| Absolute home-directory paths | PASS | All text | None found |
| Personal emails | PASS | All text | None found |
| ORCID | PASS | All text | None found |
| Personal GitHub usernames | PASS | All text | None found |
| Institution / university / acknowledgements | PASS | All text | None found |
| Personal GitHub URL | PASS | All text | README uses `<ANONYMOUS_REPOSITORY_URL>` only |
| Third-party GitHub links | ALLOWED | `LICENSE_NOTES.md`, `prepare_external_datasets.py` | RecBole / LightGCN-PyTorch **project** URLs (not personal author profile) |
| API tokens / passwords | PASS | All text + `.gitignore` | No secrets present; patterns ignored |
| WandB config | PASS | — | Not included; `wandb/` gitignored |
| Absolute paths in CSV/JSON | PASS (after fix) | `results/post_learning/steck_84.csv` | Checkpoint column reduced to `checkpoints/<basename>` |
| PDF metadata | PASS | `figures/paper/*.pdf`, supp PDF | Matplotlib creator/producer only; no author field |
| IDE / OS junk | PASS | Tree | Excluded via `.gitignore` |
| Source `.git` history | PASS | Destination only | Fresh `git init`; no source history |
| CITATION.cff authors | PASS | — | Omitted for double-blind review |
| Stale wrong methods docs | PASS | — | Excluded from copy |

## Residual notes for authors (manual)

1. Confirm no identifying text is added when pushing to the anonymous review remote.
2. After acceptance, restore `CITATION.cff` / LICENSE as appropriate.
3. Optional full-repro runners expect local `dataset/` and `runs/`; they are not required for number verification.
