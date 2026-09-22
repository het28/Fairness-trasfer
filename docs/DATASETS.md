# Datasets

Raw interaction dumps are **not** shipped in this artifact. Place datasets under a local `dataset/` directory (create it yourself) following RecBole atomic-file conventions, or adjust `data_path` in configs.

## MovieLens 1M (PRIMARY)

| Field | Value |
|---|---|
| Canonical name | MovieLens-1M |
| Source | GroupLens / RecBole dataset `ml-1m` |
| Users / items / interactions | 6,040 / 3,706 / 1,000,209 |
| Interaction semantics | Implicit positives in ECIR runs (\(c_{ui}=1\); rating column not loaded) |
| Split | RecBole user-grouped RS 80/10/10, `order: RO` |
| Raw data included? | **No** |
| How to obtain | RecBole download of MovieLens-1M, or GroupLens release processed to RecBole `.inter` |
| Prepare | Ensure `dataset/ml-1m/ml-1m.inter` exists; configs use `dataset: ml-1m` |

## Last.fm (PRIMARY)

| Field | Value |
|---|---|
| Canonical name | Last.fm (RecBole `lastfm`; HetRec2011 lineage) |
| Users / items / interactions | 1,892 / 17,632 / 92,834 |
| Interaction semantics | Listening counts in `weight`; \(c_{ui}=\log(1+w_{ui})\) |
| Split | Same RS 80/10/10 RO |
| Raw data included? | **No** |
| How to obtain | RecBole `lastfm` dataset |
| Prepare | `dataset/lastfm/` with `user_id`, `artist_id`, `weight` |

## Gowalla (TARGETED_EXTERNAL)

| Field | Value |
|---|---|
| Canonical name | Gowalla (LightGCN-preprocessed ECIR export) |
| Source | LightGCN-PyTorch `data/gowalla` (train+test merged), **not** raw SNAP alone |
| Users / items / interactions | 29,858 / 40,981 / 1,027,370 |
| Semantics | Implicit |
| Split | After merge: RecBole RS 80/10/10 RO |
| Raw data included? | **No** |
| Prepare | `python scripts/prepare_external_datasets.py` after placing LightGCN dumps (see script docstring) |

## Amazon Books (EXTERNAL_CASE_STUDY)

| Field | Value |
|---|---|
| Canonical name | Amazon Books (LightGCN-preprocessed ECIR export) |
| Source | LightGCN-PyTorch `data/amazon-book` |
| Users / items / interactions | 52,643 / 91,599 / 2,984,108 |
| Semantics | Implicit |
| Split | RecBole RS 80/10/10 RO |
| Coverage | LightGCN, seed **0** only |
| Raw data included? | **No** |

## Licensing / redistribution

MovieLens, Last.fm, Gowalla, and Amazon Books each have their own terms. This artifact does **not** redistribute raw dumps. Users must obtain data from the upstream providers / LightGCN dumps as documented above.
