# NOTE: For the anonymous artifact, prefer scripts/verify_artifact.py over re-running this historical aggregator.
#!/usr/bin/env python3
"""Build ECIR paper-ready results export from frozen artifacts only."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
FT = ROOT / "results" / "primary"
EXT = ROOT / "results" / "external"
PP = FT / "postprocessing_steck"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SHORT = {
    "weighted_bpr": "BPR",
    "weighted_neumf": "NeuMF",
    "weighted_ngcf": "NGCF",
    "weighted_lightgcn": "LightGCN",
    "weighted_itemknn": "ItemKNN",
}
DS = {"ml1m": "MovieLens-1M", "lastfm": "LastFM"}
FAM = {"legacy": "inverse-power", "kl": "minimum-perturbation KL", "baseline": "baseline"}


def mshort(m: str) -> str:
    return SHORT.get(m, str(m).replace("weighted_", ""))


def boot_slope_ci(x, y, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-12:
        return np.nan, np.nan, np.nan
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if np.std(x[idx]) < 1e-12:
            continue
        slopes.append(float(np.polyfit(x[idx], y[idx], 1)[0]))
    s = float(np.polyfit(x, y, 1)[0])
    if not slopes:
        return s, np.nan, np.nan
    return s, float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975))


def ols_design(y, X):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    n, p = X.shape
    adj = 1.0 - (1.0 - r2) * (n - 1) / max(n - p, 1) if n > p else np.nan
    return r2, adj, n, p


def frozen_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ig = out.InputGain.astype(float)
    og = out.OutputGain.astype(float)
    out["realization_ratio"] = np.where(ig.abs() > 1e-12, og / ig, np.nan)
    out["attenuated_frozen"] = (ig > 0) & (og < ig)
    out["reversed_frozen"] = (ig > 0) & (og <= 0)
    out["output_gain_zero"] = og == 0
    out["output_gain_negative"] = og < 0
    return out


def main():
    tm = pd.read_csv(FT / "transfer_metrics.csv")
    assert len(tm) == 270
    base = tm[tm.label == "baseline"].copy()
    intv = tm[tm.label != "baseline"].copy()
    assert len(base) == 30 and len(intv) == 240
    frz = json.loads((FT / "contribution_validation_summary.json").read_text())
    base_idx = base.set_index(["dataset", "model", "seed"])

    # ------------------------------------------------------------------ 1 MASTER
    master_rows = []
    for _, r in intv.iterrows():
        b = base_idx.loc[(r.dataset, r.model, r.seed)]
        master_rows.append(
            {
                "dataset": r.dataset,
                "dataset_name": DS[r.dataset],
                "model": mshort(r.model),
                "model_raw": r.model,
                "seed": int(r.seed),
                "intervention_family": FAM.get(r.family, r.family),
                "intervention_family_code": r.family,
                "intervention_parameter": r.strength,
                "label": r.label,
                "configuration_type": "intervention",
                "F_in_baseline": float(b.F_in),
                "F_in_intervention": float(r.F_in),
                "F_out_baseline": float(b.F_out),
                "F_out_intervention": float(r.F_out),
                "InputGain": float(r.InputGain),
                "OutputGain": float(r.OutputGain),
                "realization_ratio": float(r.OutputGain / r.InputGain) if abs(r.InputGain) > 1e-12 else np.nan,
                "attenuated_frozen": bool(r.InputGain > 0 and r.OutputGain < r.InputGain),
                "reversed_frozen": bool(r.InputGain > 0 and r.OutputGain <= 0),
                "NDCG10_baseline": float(b["NDCG@10"]),
                "NDCG10_intervention": float(r["NDCG@10"]),
                "relative_NDCG_loss": float(r.RelativeUtilityLoss),
                "Recall10_baseline": float(b["Recall@10"]),
                "Recall10_intervention": float(r["Recall@10"]),
                "evidence_tier": "PRIMARY",
            }
        )
    master = pd.DataFrame(master_rows)
    assert len(master) == 240
    master.to_csv(OUT / "MASTER_PRIMARY_RESULTS.csv", index=False)

    # ------------------------------------------------------------------ 2 BASELINES
    bl_rows = []
    for (ds, mod), g in base.groupby(["dataset", "model"]):
        bl_rows.append(
            {
                "dataset": ds,
                "dataset_name": DS[ds],
                "model": mshort(mod),
                "model_raw": mod,
                "n_seeds": len(g),
                "F_in_mean": float(g.F_in.mean()),
                "F_in_std": float(g.F_in.std(ddof=0)),
                "F_out_mean": float(g.F_out.mean()),
                "F_out_std": float(g.F_out.std(ddof=0)),
                "NDCG10_mean": float(g["NDCG@10"].mean()),
                "NDCG10_std": float(g["NDCG@10"].std(ddof=0)),
                "Recall10_mean": float(g["Recall@10"].mean()),
                "Recall10_std": float(g["Recall@10"].std(ddof=0)),
            }
        )
    pd.DataFrame(bl_rows).sort_values(["dataset", "model"]).to_csv(OUT / "TABLE_BASELINES.csv", index=False)

    intv = frozen_flags(intv)

    # ------------------------------------------------------------------ 3 RQ1 GLOBAL
    def rq1_block(df, scope):
        ig, og = df.InputGain.values, df.OutputGain.values
        sp = stats.spearmanr(ig, og)
        kt = stats.kendalltau(ig, og)
        b0, b1 = np.polyfit(ig, og, 1)
        _, lo, hi = boot_slope_ci(ig, og)
        r2 = 1 - np.sum((og - np.polyval([b0, b1], ig)) ** 2) / np.sum((og - og.mean()) ** 2)
        return {
            "scope": scope,
            "n": len(df),
            "mean_InputGain": float(df.InputGain.mean()),
            "median_InputGain": float(df.InputGain.median()),
            "mean_OutputGain": float(df.OutputGain.mean()),
            "median_OutputGain": float(df.OutputGain.median()),
            "spearman_rho": float(sp.correlation),
            "spearman_p": float(sp.pvalue),
            "kendall_tau": float(kt.correlation),
            "kendall_p": float(kt.pvalue),
            "gain_slope_beta1": float(b1),
            "gain_slope_intercept_beta0": float(b0),
            "gain_slope_ci_low_95": lo,
            "gain_slope_ci_high_95": hi,
            "gain_space_R2": float(r2),
            "n_OutputGain_positive": int((df.OutputGain > 0).sum()),
            "rate_OutputGain_positive": float((df.OutputGain > 0).mean()),
            "n_OutputGain_zero": int((df.OutputGain == 0).sum()),
            "n_OutputGain_negative": int((df.OutputGain < 0).sum()),
            "authoritative_note": "Gain-space descriptive; frozen M0-M2 use F_out~F_in (see TABLE_RQ2_REGRESSION)",
        }

    rq1 = pd.DataFrame([rq1_block(intv, "PRIMARY_ALL_240")])
    rq1.to_csv(OUT / "TABLE_RQ1_GLOBAL.csv", index=False)
    pd.DataFrame(
        [
            rq1_block(intv[intv.family == "legacy"], "inverse-power"),
            rq1_block(intv[intv.family == "kl"], "minimum-perturbation KL"),
        ]
    ).to_csv(OUT / "TABLE_RQ1_BY_FAMILY.csv", index=False)

    # ------------------------------------------------------------------ 4 RQ2
    def cell_row(ds, mod, g):
        s, lo, hi = boot_slope_ci(g.F_in.values, g.F_out.values)
        sp = stats.spearmanr(g.F_in, g.F_out)
        n = len(g)
        return {
            "dataset": ds,
            "dataset_name": DS[ds],
            "model": mshort(mod),
            "model_raw": mod,
            "n": n,
            "mean_InputGain": float(g.InputGain.mean()),
            "mean_OutputGain": float(g.OutputGain.mean()),
            "median_OutputGain": float(g.OutputGain.median()),
            "Fout_Fin_slope": s,
            "Fout_Fin_slope_ci_low": lo,
            "Fout_Fin_slope_ci_high": hi,
            "spearman_rho_Fin_Fout": float(sp.correlation),
            "attenuation_n": int(g.attenuated_frozen.sum()),
            "attenuation_rate": float(g.attenuated_frozen.mean()),
            "reversal_n": int(g.reversed_frozen.sum()),
            "reversal_rate": float(g.reversed_frozen.mean()),
            "mean_relative_NDCG_loss": float(g.RelativeUtilityLoss.mean()),
            "mean_NDCG10_baseline": float(
                base_idx.loc[g.apply(lambda r: (r.dataset, r.model, r.seed), axis=1).tolist()]["NDCG@10"].mean()
                if False else np.nan
            ),
        }

    cell_rows = []
    for (ds, mod), g in intv.groupby(["dataset", "model"]):
        row = cell_row(ds, mod, g)
        ndcgs = []
        for _, r in g.iterrows():
            ndcgs.append(float(base_idx.loc[(r.dataset, r.model, r.seed)]["NDCG@10"]))
        row["mean_NDCG10_baseline"] = float(np.mean(ndcgs))
        row["mean_NDCG10_intervention"] = float(g["NDCG@10"].mean())
        cell_rows.append(row)
    cells = pd.DataFrame(cell_rows).sort_values(["dataset", "model"])
    cells.to_csv(OUT / "TABLE_RQ2_CELLS.csv", index=False)

    arch_rows, ds_rows = [], []
    for mod, g in intv.groupby("model"):
        row = cell_row(g.dataset.iloc[0], mod, g)
        row["model"] = mshort(mod)
        arch_rows.append(row)
    for ds, g in intv.groupby("dataset"):
        row = cell_row(ds, g.model.iloc[0], g)
        row["dataset"] = ds
        row["dataset_name"] = DS[ds]
        ds_rows.append(row)
    pd.DataFrame(arch_rows).sort_values("model").to_csv(OUT / "TABLE_RQ2_ARCHITECTURES.csv", index=False)
    pd.DataFrame(ds_rows).sort_values("dataset").to_csv(OUT / "TABLE_RQ2_DATASETS.csv", index=False)

    n = len(intv)
    y = intv.F_out.values
    X0 = np.column_stack([np.ones(n), intv.F_in.values])
    r0, a0, n0, p0 = ols_design(y, X0)
    md = pd.get_dummies(intv.model, drop_first=True).astype(float)
    dd = pd.get_dummies(intv.dataset, drop_first=True).astype(float)
    X1 = np.column_stack([np.ones(n), intv.F_in.values, md.values, dd.values])
    r1, a1, n1, p1 = ols_design(y, X1)
    cell = intv.model.astype(str) + "|" + intv.dataset.astype(str)
    cd = pd.get_dummies(cell, drop_first=True).astype(float)
    X2 = np.column_stack([np.ones(n), intv.F_in.values, cd.values])
    r2, a2, n2, p2 = ols_design(y, X2)
    reg = pd.DataFrame(
        [
            dict(
                model_id="M0",
                formula="F_out ~ F_in",
                r2=r0,
                adjusted_r2=a0,
                n_obs=n0,
                n_params=p0,
                frozen_r2=frz["m0_r2"],
                authoritative=True,
            ),
            dict(
                model_id="M1",
                formula="F_out ~ F_in + Dataset + Model",
                r2=r1,
                adjusted_r2=a1,
                n_obs=n1,
                n_params=p1,
                frozen_r2=frz["m1_r2"],
                authoritative=True,
            ),
            dict(
                model_id="M2",
                formula="F_out ~ F_in + (Dataset×Model cell dummies, additive)",
                r2=r2,
                adjusted_r2=a2,
                n_obs=n2,
                n_params=p2,
                frozen_r2=frz["m2_r2"],
                authoritative=True,
            ),
        ]
    )
    reg.to_csv(OUT / "TABLE_RQ2_REGRESSION.csv", index=False)

    # ------------------------------------------------------------------ 5 RQ3 + UTILITY
    pos = intv[intv.InputGain > 0]
    ratio = pos.realization_ratio.dropna()

    def fail_slice(g, **kw):
        n = len(g)
        return {
            "n": n,
            "attenuation_n": int(g.attenuated_frozen.sum()),
            "attenuation_rate": float(g.attenuated_frozen.mean()) if n else np.nan,
            "reversal_n": int(g.reversed_frozen.sum()),
            "reversal_rate": float(g.reversed_frozen.mean()) if n else np.nan,
            "n_OutputGain_zero": int((g.OutputGain == 0).sum()),
            "n_OutputGain_negative": int((g.OutputGain < 0).sum()),
            "realization_ratio_mean": float(g.realization_ratio.mean()),
            "realization_ratio_median": float(g.realization_ratio.median()),
            "realization_ratio_q25": float(g.realization_ratio.quantile(0.25)),
            "realization_ratio_q75": float(g.realization_ratio.quantile(0.75)),
            **kw,
        }

    fail_rows = [fail_slice(pos, slice="PRIMARY_ALL")]
    for ds, g in pos.groupby("dataset"):
        fail_rows.append(fail_slice(g, slice="dataset", dataset=ds))
    for mod, g in pos.groupby("model"):
        fail_rows.append(fail_slice(g, slice="model", model=mshort(mod)))
    for (ds, mod), g in pos.groupby(["dataset", "model"]):
        fail_rows.append(fail_slice(g, slice="dataset_model", dataset=ds, model=mshort(mod)))
    for fam, g in pos.groupby("family"):
        fail_rows.append(fail_slice(g, slice="family", family=FAM.get(fam, fam)))
    for (fam, param), g in pos.groupby(["family", "strength"]):
        fail_rows.append(
            fail_slice(
                g,
                slice="family_strength",
                family=FAM.get(fam, fam),
                intervention_parameter=param,
            )
        )
    pd.DataFrame(fail_rows).to_csv(OUT / "TABLE_RQ3_FAILURES.csv", index=False)

    lu = intv.RelativeUtilityLoss.astype(float)
    util = {
        "n": len(intv),
        "mean_relative_NDCG_loss": float(lu.mean()),
        "median_relative_NDCG_loss": float(lu.median()),
        "q25_relative_NDCG_loss": float(lu.quantile(0.25)),
        "q75_relative_NDCG_loss": float(lu.quantile(0.75)),
        "min_relative_NDCG_loss": float(lu.min()),
        "max_relative_NDCG_loss": float(lu.max()),
        "n_loss_gt_1pct": int((lu > 0.01).sum()),
        "rate_loss_gt_1pct": float((lu > 0.01).mean()),
        "n_loss_gt_5pct": int((lu > 0.05).sum()),
        "rate_loss_gt_5pct": float((lu > 0.05).mean()),
        "n_loss_gt_10pct": int((lu > 0.10).sum()),
        "rate_loss_gt_10pct": float((lu > 0.10).mean()),
        "spearman_OutputGain_relative_NDCG_loss": float(frz["corr_util_og"]),
        "frozen_source": "contribution_validation_summary.json",
    }
    util_rows = [util]
    for (ds, mod), g in intv.groupby(["dataset", "model"]):
        util_rows.append(
            {
                **{k: v for k, v in util.items() if k in ("frozen_source",)},
                "slice": "dataset_model",
                "dataset": ds,
                "model": mshort(mod),
                "n": len(g),
                "mean_relative_NDCG_loss": float(g.RelativeUtilityLoss.mean()),
                "median_relative_NDCG_loss": float(g.RelativeUtilityLoss.median()),
                "spearman_OutputGain_relative_NDCG_loss": float(
                    stats.spearmanr(g.OutputGain, g.RelativeUtilityLoss).correlation
                ),
            }
        )
    for fam, g in intv.groupby("family"):
        util_rows.append(
            {
                "slice": "family",
                "family": FAM.get(fam, fam),
                "n": len(g),
                "mean_relative_NDCG_loss": float(g.RelativeUtilityLoss.mean()),
                "median_relative_NDCG_loss": float(g.RelativeUtilityLoss.median()),
                "spearman_OutputGain_relative_NDCG_loss": float(
                    stats.spearmanr(g.OutputGain, g.RelativeUtilityLoss).correlation
                ),
            }
        )
    pd.DataFrame(util_rows).to_csv(OUT / "TABLE_UTILITY_TRADEOFF.csv", index=False)

    # ------------------------------------------------------------------ 6-7 intervention strength / families
    str_rows = []
    for (fam, param), g in intv.groupby(["family", "strength"]):
        str_rows.append(
            {
                "intervention_family": FAM.get(fam, fam),
                "intervention_parameter": param,
                "n": len(g),
                "mean_InputGain": float(g.InputGain.mean()),
                "mean_OutputGain": float(g.OutputGain.mean()),
                "median_realization_ratio": float(g.realization_ratio.median()),
                "attenuation_rate": float(g.attenuated_frozen.mean()),
                "reversal_rate": float(g.reversed_frozen.mean()),
                "mean_relative_NDCG_loss": float(g.RelativeUtilityLoss.mean()),
                "mean_Recall_change": float((g["Recall@10"] - g.apply(lambda r: base_idx.loc[(r.dataset, r.model, r.seed)]["Recall@10"], axis=1)).mean()),
            }
        )
    pd.DataFrame(str_rows).sort_values(["intervention_family", "intervention_parameter"]).to_csv(
        OUT / "TABLE_INTERVENTION_STRENGTH.csv", index=False
    )

    fam_cmp = []
    for fam, g in intv.groupby("family"):
        fam_cmp.append(
            {
                "intervention_family": FAM.get(fam, fam),
                "n": len(g),
                "mean_InputGain": float(g.InputGain.mean()),
                "mean_OutputGain": float(g.OutputGain.mean()),
                "median_realization_ratio": float(g.realization_ratio.median()),
                "attenuation_rate": float(g.attenuated_frozen.mean()),
                "reversal_rate": float(g.reversed_frozen.mean()),
                "mean_relative_NDCG_loss": float(g.RelativeUtilityLoss.mean()),
                "mean_Recall_change": float(
                    (g["Recall@10"] - g.apply(lambda r: base_idx.loc[(r.dataset, r.model, r.seed)]["Recall@10"], axis=1)).mean()
                ),
            }
        )
    pd.DataFrame(fam_cmp).to_csv(OUT / "TABLE_INTERVENTION_FAMILIES.csv", index=False)

    # ------------------------------------------------------------------ 8 user level
    lexp = pd.read_csv(FT / "per_user_exposure_summary.csv")
    ul = lexp.copy()
    ul["model_short"] = ul.model_raw.map(mshort) if "model_raw" in ul.columns else ul.model.map(mshort)
    ul["dataset_name"] = ul.dataset.map(DS)
    ul["Q25_available"] = False  # frozen artifact has p75 not q25
    ul = ul.rename(
        columns={
            "l_exp_mean": "Lexp_mean",
            "l_exp_median": "Lexp_median",
            "l_exp_p75": "Lexp_Q75",
            "l_exp_p90": "Lexp_Q90",
            "l_exp_p95": "Lexp_Q95",
            "delta_l_exp_mean": "delta_Lexp_mean",
            "delta_F_out": "delta_F_out",
        }
    )
    ul["Lexp_Q25"] = np.nan
    ul.to_csv(OUT / "TABLE_USER_LEVEL.csv", index=False)

    # ------------------------------------------------------------------ 9 post-learning
    stage = pd.read_csv(PP / "stage_comparison_matched_utility.csv")
    stage["post_InputGain"] = 0.0
    stage.to_csv(OUT / "TABLE_POST_LEARNING.csv", index=False)
    stage.groupby(["dataset", "model", "budget"]).agg(
        mean_pre_OutputGain=("pre_OutputGain", "mean"),
        mean_post_delta_F_out=("post_delta_F_out", "mean"),
        mean_post_F_out=("post_F_out", "mean"),
        n=("seed", "count"),
    ).reset_index().to_csv(OUT / "TABLE_POST_LEARNING_AGG.csv", index=False)

    # ------------------------------------------------------------------ 10 external
    gow = pd.read_csv(FT / "gowalla_ngcf_seeds012.csv")
    amaz = pd.read_csv(EXT / "completed_runs_so_far.csv")
    ext_rows = []
    for _, r in gow.iterrows():
        ig, og = float(r.InputGain), float(r.OutputGain)
        ext_rows.append(
            {
                "dataset": "Gowalla",
                "model": "NGCF",
                "seed": int(r.seed),
                "configuration": r.label,
                "F_in": float(r.F_in),
                "F_out": float(r.F_out),
                "InputGain": ig,
                "OutputGain": og,
                "realization_ratio": og / ig if abs(ig) > 1e-12 else np.nan,
                "NDCG10": float(r.NDCG),
                "Recall10": float(r.Recall) if "Recall" in r.index else np.nan,
                "reversed_frozen": bool(ig > 0 and og <= 0),
                "attenuated_frozen": bool(ig > 0 and og < ig),
                "evidence_tier": "TARGETED_EXTERNAL",
            }
        )
    for _, r in amaz.iterrows():
        ig, og = float(r.InputGain), float(r.OutputGain)
        ext_rows.append(
            {
                "dataset": "Amazon Books",
                "model": "LightGCN",
                "seed": int(r.seed),
                "configuration": r.label,
                "F_in": float(r.F_in),
                "F_out": float(r.F_out),
                "InputGain": ig,
                "OutputGain": og,
                "realization_ratio": og / ig if abs(ig) > 1e-12 else np.nan,
                "NDCG10": float(r.NDCG),
                "Recall10": float(r.Recall) if "Recall" in r.index else np.nan,
                "reversed_frozen": bool(ig > 0 and og <= 0),
                "attenuated_frozen": bool(ig > 0 and og < ig),
                "evidence_tier": "EXTERNAL_CASE_STUDY",
                "coverage_note": "LightGCN seed 0 only; not replicated",
            }
        )
    pd.DataFrame(ext_rows).to_csv(OUT / "TABLE_EXTERNAL.csv", index=False)

    # ------------------------------------------------------------------ 11 figure CSVs
    fig1 = master[
        [
            "InputGain",
            "OutputGain",
            "dataset",
            "dataset_name",
            "model",
            "intervention_family",
            "seed",
            "intervention_parameter",
            "realization_ratio",
            "reversed_frozen",
        ]
    ].copy()
    fig1.to_csv(FIG / "FIG1_REALIZATION.csv", index=False)

    fig2 = cells[
        [
            "dataset",
            "dataset_name",
            "model",
            "Fout_Fin_slope",
            "Fout_Fin_slope_ci_low",
            "Fout_Fin_slope_ci_high",
            "mean_OutputGain",
            "reversal_rate",
            "attenuation_rate",
        ]
    ].copy()
    fig2.to_csv(FIG / "FIG2_MODEL_REALIZATION.csv", index=False)

    fig3 = master[
        [
            "intervention_family",
            "intervention_parameter",
            "InputGain",
            "OutputGain",
            "realization_ratio",
            "relative_NDCG_loss",
            "dataset",
            "model",
        ]
    ].copy()
    fig3.to_csv(FIG / "FIG3_STRENGTH_UTILITY.csv", index=False)

    stage.to_csv(FIG / "FIG4_STAGE_COMPARISON.csv", index=False)

    # ------------------------------------------------------------------ 12 plots
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )
    colors = {
        "BPR": "#1f77b4",
        "NeuMF": "#ff7f0e",
        "NGCF": "#2ca02c",
        "LightGCN": "#d62728",
        "ItemKNN": "#9467bd",
    }

    # FIG1
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for m, g in master.groupby("model"):
        ax.scatter(g.InputGain, g.OutputGain, s=18, alpha=0.75, c=colors.get(m, "#333"), label=m, edgecolors="none")
    lim = max(master.InputGain.max(), master.OutputGain.max()) * 1.05
    lo = min(master.OutputGain.min(), 0) * 1.1
    ax.plot([lo, lim], [lo, lim], "k--", lw=0.8, label="y = x")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.axvspan(0, lim, ymin=0, ymax=(0 - lo) / (lim - lo) if lim != lo else 0, color="#fee", alpha=0.25)
    ax.set_xlabel("InputGain")
    ax.set_ylabel("OutputGain")
    ax.set_xlim(lo, lim)
    ax.set_ylim(lo, lim)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_title("Primary realization (n=240)")
    fig.tight_layout()
    fig.savefig(FIG / "FIG1_REALIZATION.pdf")
    fig.savefig(FIG / "FIG1_REALIZATION.png")
    plt.close(fig)

    # FIG2
    fig2p = cells.copy()
    fig2p["cell"] = fig2p["dataset"].str.upper() + "\n" + fig2p["model"]
    fig2p = fig2p.sort_values("Fout_Fin_slope")
    ypos = np.arange(len(fig2p))
    fig, ax = plt.subplots(figsize=(3.4, 3.8))
    ax.errorbar(
        fig2p.Fout_Fin_slope,
        ypos,
        xerr=[
            fig2p.Fout_Fin_slope - fig2p.Fout_Fin_slope_ci_low,
            fig2p.Fout_Fin_slope_ci_high - fig2p.Fout_Fin_slope,
        ],
        fmt="o",
        color="#333",
        ecolor="#666",
        capsize=2,
        ms=4,
    )
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.axvline(1, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(fig2p.cell)
    ax.set_xlabel(r"Slope $F_{\mathrm{out}} \sim F_{\mathrm{in}}$")
    ax.set_title("Dataset × model realization")
    fig.tight_layout()
    fig.savefig(FIG / "FIG2_MODEL_REALIZATION.pdf")
    fig.savefig(FIG / "FIG2_MODEL_REALIZATION.png")
    plt.close(fig)

    # FIG3
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for fam, g in master.groupby("intervention_family"):
        ax.scatter(g.InputGain, g.relative_NDCG_loss, s=16, alpha=0.6, label=fam)
    ax.set_xlabel("InputGain")
    ax.set_ylabel("Relative NDCG loss")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Intervention strength vs utility cost")
    fig.tight_layout()
    fig.savefig(FIG / "FIG3_STRENGTH_UTILITY.pdf")
    fig.savefig(FIG / "FIG3_STRENGTH_UTILITY.png")
    plt.close(fig)

    # FIG4
    agg = stage.groupby(["model", "budget"]).agg(
        pre=("pre_OutputGain", "mean"),
        post=("post_delta_F_out", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    x = np.arange(len(agg))
    w = 0.35
    ax.bar(x - w / 2, agg.pre, width=w, label="Pre-learning OutputGain", color="#4c72b0")
    ax.bar(x + w / 2, agg.post, width=w, label="Post-learning ΔF_out", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n{b*100:.1f}%" for m, b in zip(agg.model, agg.budget)], fontsize=7)
    ax.set_ylabel("Exposure improvement")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Matched-utility stage contrast")
    fig.tight_layout()
    fig.savefig(FIG / "FIG4_STAGE_COMPARISON.pdf")
    fig.savefig(FIG / "FIG4_STAGE_COMPARISON.png")
    plt.close(fig)

    print("BUILD OK", OUT)
    return master, reg, frz


if __name__ == "__main__":
    main()
