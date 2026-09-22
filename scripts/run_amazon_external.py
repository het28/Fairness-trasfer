"""ANONYMOUS ARTIFACT NOTE
This runner is included for optional FULL reproduction.
Repository-relative paths may still reference historical layout strings in constants.
For reviewer verification of reported numbers, use scripts/verify_artifact.sh instead of this file.
Scientific defaults (grids, seeds, formulas) must not be changed.
"""
#!/usr/bin/env python3
"""Minimal external validation matrix (90 runs) for fairness transfer.

Writes ONLY under research_ecir2027/results/ecir2027/external_transfer/.
Requires meg_rw_save_calibration_info so F_in is logged before training.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
os.environ["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")

from training.run_experiment import run_experiment

OUT = ROOT / "runs" / "external" / "amazon"
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2").split()]
DATASETS = os.environ.get("DATASETS", "amazonbooks_ecir gowalla_ecir").split()
MODELS = os.environ.get("MODELS", "weighted_lightgcn weighted_ngcf weighted_bpr").split()

DATASET_CFG = {
    "amazonbooks_ecir": {
        "yaml": "configs/external/amazon_books_{stem}.yaml",
        "short": "amazonbooks_ecir",
        "recbole": "amazon-books-ecir",
        "multiply_c_ui": True,
        "c_ui_transform": "identity",  # implicit ones
    },
    "gowalla_ecir": {
        "yaml": "configs/external/gowalla_{stem}.yaml",
        "short": "gowalla_ecir",
        "recbole": "gowalla-ecir",
        "multiply_c_ui": True,
        "c_ui_transform": "identity",
    },
}
MODEL_STEM = {
    "weighted_lightgcn": "lightgcn",
    "weighted_ngcf": "ngcf",
    "weighted_bpr": "bpr",
}


def configs(label: str) -> dict:
    if label == "baseline":
        return {"meg_rw_alpha": 0.0, "meg_cal_mode": "", "meg_cal_lambda_r": 0.0}
    if label.startswith("legacy_a"):
        a = float(label.replace("legacy_a", ""))
        return {"meg_rw_alpha": a, "meg_cal_mode": "legacy_inverse_power", "meg_cal_lambda_r": 0.0}
    if label.startswith("kl_lam"):
        lam = float(label.replace("kl_lam", ""))
        return {
            "meg_rw_alpha": 0.0,
            "meg_cal_mode": "kl_projection",
            "meg_cal_lambda_r": lam,
            "meg_cal_target": "catalog",
        }
    raise ValueError(label)


LABELS = ["baseline", "legacy_a0.2", "legacy_a0.8", "kl_lam0.1", "kl_lam2.0"]


def _done(path: Path) -> bool:
    return (
        (path / "fairness_report.json").exists()
        and (path / "metrics.json").exists()
        and (path / "input_calibration.json").exists()
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for ds in DATASETS:
        meta = DATASET_CFG[ds]
        for model in MODELS:
            stem = MODEL_STEM[model]
            yfiles = [
                str(ROOT / "configs/primary/base.yaml"),
                str(ROOT / meta["yaml"].format(stem=stem)),
            ]
            for seed in SEEDS:
                for label in LABELS:
                    report = OUT / ds / model / f"seed_{seed}" / label
                    if _done(report):
                        print(f"SKIP {ds}/{model}/seed{seed}/{label}", flush=True)
                        n += 1
                        continue
                    report.mkdir(parents=True, exist_ok=True)
                    cfg = {
                        "seed": seed,
                        "dataset": meta["recbole"],
                        "backbone": model,
                        "dataset_short": meta["short"],
                        "model_short": model,
                        "run_label": label,
                        "meg_rw_multiply_c_ui": meta["multiply_c_ui"],
                        "meg_rw_c_ui_transform": meta["c_ui_transform"],
                        "meg_rw_save_calibration_info": True,
                        "use_gpu": False,
                        "checkpoint_dir": str(OUT / "_checkpoints"),
                    }
                    cfg.update(configs(label))
                    print(f"RUN  {ds}/{model}/seed{seed}/{label}", flush=True)
                    run_experiment(
                        config_file_list=yfiles,
                        config_dict=cfg,
                        fairness_audit=True,
                        fairness_report_dir=str(report),
                        fairness_save_eval_inputs=True,
                    )
                    assert _done(report), f"incomplete {report}"
                    n += 1
    print(f"PHASE_COMPLETE external_transfer jobs_touched={n}", flush=True)


if __name__ == "__main__":
    main()
