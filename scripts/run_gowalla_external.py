"""ANONYMOUS ARTIFACT NOTE
This runner is included for optional FULL reproduction.
Repository-relative paths may still reference historical layout strings in constants.
For reviewer verification of reported numbers, use scripts/verify_artifact.sh instead of this file.
Scientific defaults (grids, seeds, formulas) must not be changed.
"""
#!/usr/bin/env python3
"""Gowalla NGCF replication: seeds 1 and 2 only.

Authorised: baseline, legacy_a0.8, kl_lam0.1 × seeds {1, 2} = 6 runs.
Same pipeline semantics and pre-training F_in logging as seed 0.
Does not launch seed 0, BPR, LightGCN, Amazon, or any other config.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from training.run_experiment import run_experiment

OUT = ROOT / "runs" / "external" / "gowalla"

DATASET_RECBOLE = "gowalla-ecir"
DATASET_SHORT = "gowalla_ecir"
MODEL = "weighted_ngcf"
SEEDS = [1, 2]
AUTHORISED_LABELS = ["baseline", "legacy_a0.8", "kl_lam0.1"]

YFILES = [
    str(ROOT / "configs/primary/base.yaml"),
    str(ROOT / "configs/external/gowalla_ngcf.yaml"),
]


def label_cfg(label: str) -> dict:
    if label == "baseline":
        return {"meg_rw_alpha": 0.0, "meg_cal_mode": "", "meg_cal_lambda_r": 0.0}
    if label == "legacy_a0.8":
        return {"meg_rw_alpha": 0.8, "meg_cal_mode": "legacy_inverse_power", "meg_cal_lambda_r": 0.0}
    if label == "kl_lam0.1":
        return {
            "meg_rw_alpha": 0.0,
            "meg_cal_mode": "kl_projection",
            "meg_cal_lambda_r": 0.1,
            "meg_cal_target": "catalog",
        }
    raise ValueError(f"Unauthorised label: {label}")


def done(path: Path) -> bool:
    return (
        (path / "fairness_report.json").exists()
        and (path / "metrics.json").exists()
        and (path / "input_calibration.json").exists()
    )


def tv(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / max(a.sum(), 1e-12)
    b = b / max(b.sum(), 1e-12)
    return float(0.5 * np.abs(a - b).sum())


def verify_fin(run_dir: Path) -> tuple[bool, str]:
    cal = run_dir / "input_calibration.json"
    if not cal.exists():
        return False, "MISSING input_calibration.json"
    d = json.loads(cal.read_text())
    q = d.get("q_in_calibrated", d.get("q_in"))
    T = d.get("T")
    logged = d.get("F_in_calibrated", d.get("F_in"))
    if q is None or T is None or logged is None:
        return False, "input_calibration.json missing q_in/T/F_in"
    recon = tv(q, T)
    disc = abs(recon - float(logged))
    if disc > 1e-6:
        return False, f"F_in_logged={logged} recon={recon} disc={disc} > tolerance"
    return True, f"OK F_in={logged} recon={recon} disc={disc}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    print("=== GOWALLA NGCF REPLICATION — SEEDS 1 AND 2 ONLY ===", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(f"Labels: {AUTHORISED_LABELS}", flush=True)

    for seed in SEEDS:
        for label in AUTHORISED_LABELS:
            run_dir = OUT / DATASET_SHORT / MODEL / f"seed_{seed}" / label
            print(f"\nRUN  {DATASET_SHORT}/{MODEL}/seed_{seed}/{label}", flush=True)

            if done(run_dir):
                ok, msg = verify_fin(run_dir)
                status = "CACHED_OK" if ok else "CACHED_FIN_ISSUE"
                print(f"  SKIP (already done) — {msg}", flush=True)
                results.append({"seed": seed, "label": label, "status": status, "fin_msg": msg})
                if not ok:
                    print("  STOPPING: F_in issue on cached run. Not using as evidence.", flush=True)
                continue

            run_dir.mkdir(parents=True, exist_ok=True)
            cfg = {
                "seed": seed,
                "dataset": DATASET_RECBOLE,
                "backbone": MODEL,
                "dataset_short": DATASET_SHORT,
                "model_short": MODEL,
                "run_label": label,
                "meg_rw_multiply_c_ui": True,
                "meg_rw_c_ui_transform": "identity",
                "meg_rw_save_calibration_info": True,
                "use_gpu": False,
                "checkpoint_dir": str(OUT / "_checkpoints_ngcf_repl"),
            }
            cfg.update(label_cfg(label))

            t0 = time.time()
            try:
                run_experiment(
                    config_file_list=YFILES,
                    config_dict=cfg,
                    fairness_audit=True,
                    fairness_report_dir=str(run_dir),
                    fairness_save_eval_inputs=True,
                )
                elapsed = time.time() - t0
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  ERROR after {elapsed:.0f}s: {e}", flush=True)
                results.append({
                    "seed": seed, "label": label, "status": "ERROR",
                    "error": str(e), "elapsed_s": round(elapsed),
                })
                continue

            if not done(run_dir):
                print(f"  FAILED: output files missing after {elapsed:.0f}s", flush=True)
                results.append({
                    "seed": seed, "label": label,
                    "status": "FAILED_MISSING_FILES", "elapsed_s": round(elapsed),
                })
                continue

            ok, msg = verify_fin(run_dir)
            if not ok:
                print(f"  STOP: {msg} — do not use as evidence", flush=True)
                results.append({
                    "seed": seed, "label": label, "status": "FIN_DISCREPANCY",
                    "fin_msg": msg, "elapsed_s": round(elapsed),
                })
                continue

            print(f"  OK ({elapsed:.0f}s) — {msg}", flush=True)
            results.append({
                "seed": seed, "label": label, "status": "OK",
                "fin_msg": msg, "elapsed_s": round(elapsed),
            })

    print("\n=== SUMMARY ===", flush=True)
    for r in results:
        print(json.dumps(r), flush=True)

    log = OUT / "gowalla_ngcf_replication_summary.json"
    log.write_text(json.dumps({"runs": results, "n_authorised": 6}, indent=2))
    print(f"\nSummary written to {log}", flush=True)
    print("PHASE_COMPLETE gowalla_ngcf_replication", flush=True)


if __name__ == "__main__":
    main()
