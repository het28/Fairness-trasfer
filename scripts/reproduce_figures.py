#!/usr/bin/env python3
"""Regenerate paper figures from frozen figure-data CSVs (no training)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/primary/figure_data"
OUT = ROOT / "figures/paper"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe–Ito (colorblind-safe); consistent with paper figures
MODEL_COLOR = {
    "ItemKNN": "#0072B2",
    "BPR": "#E69F00",
    "NeuMF": "#CC79A7",
    "LightGCN": "#009E73",
    "NGCF": "#D55E00",
    "itemknn": "#0072B2",
    "bpr": "#E69F00",
    "neumf": "#CC79A7",
    "lightgcn": "#009E73",
    "ngcf": "#D55E00",
    "weighted_itemknn": "#0072B2",
    "weighted_bpr": "#E69F00",
    "weighted_neumf": "#CC79A7",
    "weighted_lightgcn": "#009E73",
    "weighted_ngcf": "#D55E00",
}


def _short_model(m: str) -> str:
    m = str(m).replace("weighted_", "")
    return {
        "itemknn": "ItemKNN",
        "bpr": "BPR",
        "neumf": "NeuMF",
        "lightgcn": "LightGCN",
        "ngcf": "NGCF",
    }.get(m.lower(), m)


def fig_realization():
    df = pd.read_csv(DATA / "fig_realization.csv")
    # flexible column names
    xcol = "InputGain" if "InputGain" in df.columns else [c for c in df.columns if "Input" in c][0]
    ycol = "OutputGain" if "OutputGain" in df.columns else [c for c in df.columns if "Output" in c][0]
    dscol = "dataset" if "dataset" in df.columns else "Dataset"
    mcol = "model" if "model" in df.columns else "Model"
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for _, r in df.iterrows():
        ds = str(r[dscol]).lower()
        marker = "o" if "ml" in ds or "movie" in ds else "s"
        model = _short_model(r[mcol])
        ax.scatter(
            r[xcol],
            r[ycol],
            c=MODEL_COLOR.get(model, "#333333"),
            marker=marker,
            s=18,
            alpha=0.65,
            linewidths=0.3,
            edgecolors="white",
        )
    lim = max(df[xcol].max(), df[ycol].max()) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", color="0.4", lw=0.9, zorder=0)
    ax.axhline(0, ls=":", color="0.5", lw=0.8, zorder=0)
    ax.set_xlabel("InputGain")
    ax.set_ylabel("OutputGain")
    ax.set_xlim(left=-0.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig_realization_repro.pdf")
    fig.savefig(OUT / "fig_realization_repro.png", dpi=300)
    plt.close(fig)
    print("wrote", OUT / "fig_realization_repro.pdf")


def fig_cell_slopes():
    df = pd.read_csv(DATA / "fig_cell_slopes.csv")
    # expect slope + CI columns
    slope = [c for c in df.columns if "slope" in c.lower() and "ci" not in c.lower()][0]
    lo = [c for c in df.columns if "ci_low" in c.lower() or c.endswith("_low")][0]
    hi = [c for c in df.columns if "ci_high" in c.lower() or c.endswith("_high")][0]
    df = df.sort_values(slope)
    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    y = np.arange(len(df))
    for i, (_, r) in enumerate(df.iterrows()):
        model = _short_model(r.get("model", r.get("model_raw", "")))
        ds = str(r.get("dataset", r.get("dataset_name", ""))).lower()
        marker = "o" if "ml" in ds else "s"
        ax.errorbar(
            r[slope],
            i,
            xerr=[[r[slope] - r[lo]], [r[hi] - r[slope]]],
            fmt=marker,
            color=MODEL_COLOR.get(model, "#333"),
            markersize=5,
            capsize=2,
            elinewidth=0.8,
        )
        label = f"{r.get('dataset_name', r.get('dataset', ds))} — {model}"
        # ytick later
    ax.axvline(0, ls=":", color="0.5", lw=0.8)
    ax.axvline(1, ls="--", color="0.45", lw=0.8)
    labels = []
    for _, r in df.iterrows():
        ds = r.get("dataset_name", r.get("dataset", ""))
        model = _short_model(r.get("model", r.get("model_raw", "")))
        labels.append(f"{ds} — {model}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Realization slope")
    fig.tight_layout()
    fig.savefig(OUT / "fig_cell_slopes_repro.pdf")
    fig.savefig(OUT / "fig_cell_slopes_repro.png", dpi=300)
    plt.close(fig)
    print("wrote", OUT / "fig_cell_slopes_repro.pdf")


def fig_stage():
    df = pd.read_csv(DATA / "fig_stage.csv")
    budget_col = "budget"
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    cell_styles = {
        ("ml1m", "lightgcn"): ("#009E73", "o"),
        ("ml1m", "ngcf"): ("#D55E00", "^"),
        ("lastfm", "lightgcn"): ("#0072B2", "s"),
        ("lastfm", "ngcf"): ("#E69F00", "D"),
    }
    for (ds, model), g in df.groupby(["dataset", "model"]):
        key = (str(ds).lower(), str(model).lower().replace("weighted_", ""))
        color, marker = cell_styles.get(key, ("#333333", "o"))
        agg = g.groupby(budget_col)[["pre_OutputGain", "post_delta_F_out"]].mean().reset_index()
        agg = agg.sort_values(budget_col)
        ax.plot(
            agg[budget_col],
            agg["pre_OutputGain"],
            ls="--",
            marker=marker,
            fillstyle="none",
            color=color,
            lw=1.0,
            markersize=5,
        )
        ax.plot(
            agg[budget_col],
            agg["post_delta_F_out"],
            ls="-",
            marker=marker,
            color=color,
            lw=1.0,
            markersize=5,
        )
    ax.axhline(0, ls=":", color="0.5", lw=0.8)
    ax.set_xlabel("Matched relative utility-loss budget")
    ax.set_ylabel("Output exposure improvement")
    fig.tight_layout()
    fig.savefig(OUT / "fig_stage_repro.pdf")
    fig.savefig(OUT / "fig_stage_repro.png", dpi=300)
    plt.close(fig)
    print("wrote", OUT / "fig_stage_repro.pdf")


def main():
    fig_realization()
    fig_cell_slopes()
    fig_stage()
    print("Frozen camera-ready PDFs remain in figures/paper/fig_*.pdf")
    print("Repro outputs written as figures/paper/*_repro.pdf")


if __name__ == "__main__":
    main()
