# Methods (implementation-faithful)

This document matches the repository implementation and the methods reproducibility audit. It intentionally **does not** repeat stale claims (raw-rating \(c_{ui}\) on ML-1M, leave-one-out splits, \(\mathrm{KL}(T\|q)\), scipy KL solver, 11-value Steck grid, or weighted positive sampling for BPR/NeuMF).

## Popularity groups

Items are sorted by **descending training interaction count** (stable argsort). The catalogue is partitioned into four groups with fractions

\[
(0.1,\ 0.2,\ 0.3,\ 0.4)
\]

(Head, UpperMid, LowerMid, Tail). Group sizes use `int(round(f * n_items))` with the last group receiving the remainder (`src/meg_rw/grouping.py`).

Groups are computed on the **train split only** and are **not** changed by interventions.

## Target \(T\)

\[
T_g = \frac{\#\{i : g(i)=g\}}{n_{\text{items}}}
\]

(empirical catalogue share; renormalized). \(T\) is approximately \((0.10,0.20,0.30,0.40)\) after integer rounding, but is **not** hard-coded (`src/meg_rw/statistics.py`).

## Effective interaction weights \(c_{ui}\)

Final edge/sample weight:

\[
\tilde w_{ui} = \phi_{g(i)}\, c_{ui}
\]

(semantic factor disabled in reported runs).

| Dataset | Effective \(c_{ui}\) |
|---|---|
| MovieLens-1M | \(c_{ui}=1\) (rating field not loaded in ECIR configs) |
| Last.fm | \(c_{ui}=\log(1+w_{ui})\) on listening counts |

Code: `src/cikm_train/recbole_dataset.py` (`_extract_c_ui`, `inject_meg_rw_into_train_dataset`).

## Inverse-power intervention

Dominance \(D_g = M_g / (C_g+\varepsilon)\) with \(\varepsilon=10^{-8}\).

\[
\phi_g = (D_g+\varepsilon)^{-\alpha},\qquad \alpha\in\{0.1,0.2,0.4,0.8\}
\]

No φ mean-normalization in primary runs (`meg_rw_normalize_phi: false`).

## Minimum-perturbation KL intervention

With \(u_g=\log w_g\) and \(q_g \propto M_g w_g\):

\[
\min_u\ \mathrm{KL}(q(w)\,\|\,T) + \lambda_R \sum_g M_g u_g^2,
\qquad \lambda_R\in\{0.1,0.5,2.0,8.0\}.
\]

Solver: fixed-step gradient descent (`lr=0.05`, `max_iter=500`, early stop on loss change); then rescale \(w\) to mean 1 (`src/meg_rw/calibration.py`).

## Exposure and gains

\[
F = \tfrac12 \|\mathrm{distribution}-T\|_1
\]

\[
\mathrm{InputGain}=F_{\mathrm{in}}^{\mathrm{base}}-F_{\mathrm{in}},\quad
\mathrm{OutputGain}=F_{\mathrm{out}}^{\mathrm{base}}-F_{\mathrm{out}}
\]

\[
\Gamma = \mathrm{OutputGain}/\mathrm{InputGain}
\]

**Operational reversal:** \(\mathrm{InputGain}>0\) and \(\mathrm{OutputGain}\le 0\).

**Attenuation (frozen rate claim):** \(\Gamma < 1\).

## Utility

Relative NDCG@10 loss vs same-seed baseline:

\[
\frac{\mathrm{NDCG}_0-\mathrm{NDCG}}{\mathrm{NDCG}_0}.
\]

## User-level exposure

\[
L_{\exp}(u)=\tfrac12\|E_u-T\|_1\in[0,1]
\]

where \(E_u\) is the user’s Top-\(K\) group-share vector (`src/cikm_eval/exposure.py`).

## Evaluation protocol

- Split: RecBole RS `[0.8,0.1,0.1]`, `order: RO`, `group_by: user`
- Full-sort ranking; Top-\(K=10\)
- Binary NDCG relevance from test positives
- \(E_{\mathrm{out}}\): unweighted frequency of groups across Top-10 slots

## Post-learning (adapted Steck)

Greedy list construction with catalogue-share target \(T\):

\[
(1-\lambda)\,\mathrm{rel}_{\mathrm{norm}} - \lambda\,\mathrm{KL}(T\|q^{\sim})
\]

\(\lambda\in\{0,0.1,0.25,0.5,0.75,0.9,1.0\}\) (84 runs → 48 matched utility comparisons). `InputGain=0` by construction.

## Models

Weighted RecBole wrappers in `src/recbole_ext/`. Hyperparameters are fixed YAML defaults (no ECIR tuning sweep). Primary training is CPU-only.
