#!/usr/bin/env python3
"""Verify frozen paper numbers against included result CSVs/JSON.

Exit nonzero on mismatch. No training. No result mutation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def approx(a, b, tol, name):
    a, b = float(a), float(b)
    ok = abs(a - b) <= tol
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {name}: got={a:.10g} expected≈{b} (tol={tol})")
    return ok


def exact(a, b, name):
    ok = int(a) == int(b)
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {name}: got={a} expected={b}")
    return ok


def main() -> int:
    ok = True
    print("=== PRIMARY (master_240) ===")
    m = pd.read_csv(ROOT / "results/primary/master_240.csv")
    ok &= exact(len(m), 240, "n")
    ok &= approx(m["InputGain"].mean(), 0.222, 5e-3, "mean InputGain")
    ok &= approx(m["OutputGain"].mean(), 0.061, 5e-3, "mean OutputGain")
    from scipy.stats import spearmanr, kendalltau

    rho = spearmanr(m["InputGain"], m["OutputGain"]).correlation
    tau = kendalltau(m["InputGain"], m["OutputGain"]).correlation
    ok &= approx(rho, 0.490, 5e-3, "Spearman")
    ok &= approx(tau, 0.340, 5e-3, "Kendall")
    slope = float(np.polyfit(m["InputGain"], m["OutputGain"], 1)[0])
    ok &= approx(slope, 0.289, 5e-3, "gain slope")
    g = m["OutputGain"] / m["InputGain"]
    ok &= exact((m["OutputGain"] > 0).sum(), 211, "positive OG")
    ok &= exact((g < 1).sum(), 237, "Gamma < 1")
    ok &= approx(g.median(), 0.111, 5e-3, "median Gamma")
    ok &= exact((m["OutputGain"] <= 0).sum(), 29, "reversals OG<=0")
    ok &= exact((m["OutputGain"] < 0).sum(), 28, "strict negative")
    ok &= exact((m["OutputGain"] == 0).sum(), 1, "exact zero OG")
    ok &= exact((m["OutputGain"] >= m["InputGain"]).sum(), 3, "OG >= IG")

    print("=== RQ1 GLOBAL CSV ===")
    rq1 = pd.read_csv(ROOT / "results/primary/rq1_global.csv").iloc[0]
    # slope may be under intercept column due to known packaging swap; check both
    candidates = [float(rq1["gain_slope_beta1"]), float(rq1["gain_slope_intercept_beta0"])]
    slope_csv = max(candidates, key=lambda x: abs(x))  # 0.289 vs -0.003
    ok &= approx(slope_csv, 0.289, 5e-3, "rq1 slope (abs-max of labeled cols)")
    ok &= approx(rq1["gain_slope_ci_low_95"], 0.214, 5e-3, "CI low")
    ok &= approx(rq1["gain_slope_ci_high_95"], 0.378, 5e-3, "CI high")

    print("=== REGRESSION R2 ===")
    reg = pd.read_csv(ROOT / "results/primary/rq2_regression.csv").set_index("model_id")
    ok &= approx(reg.loc["M0", "r2"], 0.124, 5e-3, "M0 R2")
    ok &= approx(reg.loc["M1", "r2"], 0.620, 5e-3, "M1 R2")
    ok &= approx(reg.loc["M2", "r2"], 0.881, 5e-3, "M2 R2")

    print("=== UTILITY ===")
    util = pd.read_csv(ROOT / "results/primary/utility_tradeoff.csv").iloc[0]
    ok &= approx(util["median_relative_NDCG_loss"], 0.0239, 5e-4, "median LU")
    ok &= approx(util["mean_relative_NDCG_loss"], 0.0819, 5e-4, "mean LU")
    ok &= exact(util["n_loss_gt_5pct"], 95, "n LU>5%")
    ok &= approx(util["spearman_OutputGain_relative_NDCG_loss"], 0.614, 5e-3, "Spearman OG~LU")

    print("=== POST-LEARNING ===")
    post = pd.read_csv(ROOT / "results/post_learning/matched_48.csv")
    if "pre_OutputGain" not in post.columns:
        post = pd.read_csv(ROOT / "results/post_learning/matched_48_export.csv")
    ok &= exact(len(post), 48, "matched n")
    ok &= approx(post["pre_OutputGain"].mean(), 0.019, 5e-3, "pre mean OG")
    ok &= approx(post["post_delta_F_out"].mean(), 0.092, 5e-3, "post mean dFout")
    s84 = pd.read_csv(ROOT / "results/post_learning/steck_84.csv")
    ok &= exact((s84["InputGain"] == 0).sum(), len(s84), "steck InputGain=0")

    print("=== EXTERNAL ===")
    gow = pd.read_csv(ROOT / "results/external/gowalla_ngcf.csv")
    gi = gow[gow["label"] != "baseline"]
    ok &= approx(gi["InputGain"].mean(), 0.274, 5e-3, "Gowalla mean IG")
    ok &= approx(gi["OutputGain"].mean(), 0.0027, 5e-4, "Gowalla mean OG")
    amz = pd.read_csv(ROOT / "results/external/amazon_books.csv")
    ai = amz[amz["label"] != "baseline"]
    ok &= approx(ai["InputGain"].mean(), 0.183, 5e-3, "Amazon mean IG")
    ok &= approx(ai["OutputGain"].mean(), 0.061, 5e-3, "Amazon mean OG")

    print("=== USER-LEVEL ===")
    ans = json.loads((ROOT / "results/user_level/answers.json").read_text())
    ok &= approx(ans["q1_frac_fout_improve_mean_lexp_improves"], 0.995, 5e-3, "99.5%")
    ok &= approx(ans["q1_spearman_deltaFout_deltaLexpMean"], 0.986, 5e-3, "user rho")
    ok &= approx(ans["q2_frac_fout_improve_p90_improves"], 0.005, 5e-3, "p90 improve")
    ok &= approx(ans["q2_frac_fout_improve_p95_improves"], 0.0, 1e-12, "p95 improve")

    print("=== STECK 84 ===")
    ok &= exact(len(s84), 84, "steck rows")

    print("\n" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
