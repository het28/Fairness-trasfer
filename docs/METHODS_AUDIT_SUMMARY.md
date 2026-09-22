# Methods audit summary (anonymous)

Condensed from the source methods reproducibility audit. Use with `docs/METHODS.md`.

## Resolved methodology (authoritative)

- ML-1M: \(c_{ui}=1\) (rating not loaded)
- LastFM: \(c_{ui}=\log(1+w_{ui})\)
- Split: RecBole RS 80/10/10, `order: RO`, `group_by: user`
- Groups: descending train degree; catalogue fractions 10/20/30/40%
- \(T\): empirical catalogue share
- KL: \(\mathrm{KL}(q\|T)+\lambda_R\sum_g M_g(\log w_g)^2\)
- Primary grid: \(\alpha\in\{0.1,0.2,0.4,0.8\}\), \(\lambda_R\in\{0.1,0.5,2.0,8.0\}\) → 270 configs
- Post-learning λ: \(\{0,0.1,0.25,0.5,0.75,0.9,1.0\}\) → 84 → 48 matched
- Primary training: CPU

## Do not use (stale)

- ML-1M \(c_{ui}=\) raw rating
- Leave-one-out split
- \(\mathrm{KL}(T\|q)\) / scipy KL solver
- 11-value Steck grid / 132 runs
- Weighted positive sampling description for BPR/NeuMF
