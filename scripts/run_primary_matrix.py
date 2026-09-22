"""ANONYMOUS ARTIFACT NOTE
This runner is included for optional FULL reproduction.
Repository-relative paths may still reference historical layout strings in constants.
For reviewer verification of reported numbers, use scripts/verify_artifact.sh instead of this file.
Scientific defaults (grids, seeds, formulas) must not be changed.
"""
#!/usr/bin/env python3
"""Primary matrix runner — writes under runs/primary/.

Phases:
  baselines  — α=0 / no calibration, all dataset×model×seed
  legacy     — inverse-power α grid
  kl         — KL projection λ_R grid
  all        — baselines then legacy then kl

Example:
  PYTHONPATH=src PHASE=baselines DATASETS=ml1m MODELS=weighted_lightgcn SEEDS='0' \\
    python scripts/run_primary_matrix.py
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

OUT = ROOT / "runs" / "primary"
PHASE = os.environ.get("PHASE", "baselines").lower()
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2").split()]
DATASETS = os.environ.get("DATASETS", "ml1m lastfm").split()
MODELS = os.environ.get(
    "MODELS",
    "weighted_lightgcn weighted_ngcf weighted_bpr weighted_neumf weighted_itemknn",
).split()
LEGACY_ALPHAS = [float(x) for x in os.environ.get("LEGACY_ALPHAS", "0.1 0.2 0.4 0.8").split()]
KL_LAMBDAS = [float(x) for x in os.environ.get("KL_LAMBDAS", "0.1 0.5 2.0 8.0").split()]
EPOCHS = os.environ.get("EPOCHS")  # optional override; default from yaml

DATASET_CFG = {
    "ml1m": {
        "yaml": "configs/primary/ml1m_{stem}.yaml",
        "short": "ml1m",
        "multiply_c_ui": True,
        "c_ui_transform": "identity",  # implicit ones
    },
    "lastfm": {
        "yaml": "configs/primary/lastfm_{stem}.yaml",
        "short": "lastfm",
        "multiply_c_ui": True,
        "c_ui_transform": "log1p",
    },
}

MODEL_STEM = {
    "weighted_lightgcn": "lightgcn",
    "weighted_ngcf": "ngcf",
    "weighted_bpr": "bpr",
    "weighted_neumf": "neumf",
    "weighted_itemknn": "itemknn",
}


def _base_files(dataset: str, model: str) -> list[str]:
    stem = MODEL_STEM[model]
    y = DATASET_CFG[dataset]["yaml"].format(stem=stem)
    return [str(ROOT / "configs/primary/base.yaml"), str(ROOT / y)]


def _common_cfg(dataset: str, model: str, seed: int, label: str) -> dict:
    meta = DATASET_CFG[dataset]
    cfg = {
        "seed": seed,
        "backbone": model,
        "dataset_short": meta["short"],
        "model_short": model,
        "run_label": label,
        "meg_rw_multiply_c_ui": meta["multiply_c_ui"],
        "meg_rw_c_ui_transform": meta["c_ui_transform"],
        "use_gpu": False,
                "checkpoint_dir": str(OUT / "_checkpoints"),
    }
    if EPOCHS:
        cfg["epochs"] = int(EPOCHS)
        cfg["stopping_step"] = int(EPOCHS)
    return cfg


def _done(path: Path) -> bool:
    return (path / "fairness_report.json").exists() and (path / "metrics.json").exists()


def _count_done() -> int:
    return sum(
        1
        for p in OUT.glob("**/fairness_report.json")
        if "_checkpoints" not in str(p)
    )


def _write_status(phase: str, current: str) -> None:
    done = _count_done()
    lines = [
        f"=== PRIMARY MATRIX {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===",
        f"phase: {phase}",
        f"completed_reports: {done}",
        f"current: {current}",
        "",
        "done_list:",
    ]
    for p in sorted(OUT.glob("**/fairness_report.json")):
        if "_checkpoints" in str(p):
            continue
        rel = p.relative_to(OUT).parent.as_posix()
        lines.append(f"  {rel}")
    (OUT / "status.txt").write_text("\n".join(lines) + "\n")


def _run(dataset: str, model: str, seed: int, label: str, extra: dict, phase: str = "baselines") -> None:
    report = OUT / dataset / model / f"seed_{seed}" / label
    tag = f"{dataset}/{model}/seed{seed}/{label}"
    if _done(report):
        print(f"SKIP {tag}", flush=True)
        _write_status(phase, f"SKIP {tag}")
        return
    report.mkdir(parents=True, exist_ok=True)
    cfg = _common_cfg(dataset, model, seed, label)
    cfg.update(extra)
    print(f"RUN  {tag}", flush=True)
    _write_status(phase, f"RUN  {tag}")
    run_experiment(
        config_file_list=_base_files(dataset, model),
        config_dict=cfg,
        fairness_audit=True,
        fairness_report_dir=str(report),
        fairness_save_eval_inputs=True,
    )
    assert _done(report), f"missing artifacts in {report}"
    print(f"DONE {tag}  completed_reports={_count_done()}", flush=True)
    _write_status(phase, f"DONE {tag}")


def jobs_baselines():
    for ds in DATASETS:
        for model in MODELS:
            for seed in SEEDS:
                yield ds, model, seed, "baseline", {
                    "meg_rw_alpha": 0.0,
                    "meg_cal_mode": "",
                }


def jobs_legacy():
    for ds in DATASETS:
        for model in MODELS:
            for seed in SEEDS:
                for a in LEGACY_ALPHAS:
                    yield ds, model, seed, f"legacy_a{a}", {
                        "meg_rw_alpha": a,
                        "meg_cal_mode": "legacy_inverse_power",
                    }


def jobs_kl():
    for ds in DATASETS:
        for model in MODELS:
            for seed in SEEDS:
                for lam in KL_LAMBDAS:
                    yield ds, model, seed, f"kl_lam{lam}", {
                        "meg_rw_alpha": 0.0,
                        "meg_cal_mode": "kl_projection",
                        "meg_cal_lambda_r": lam,
                        "meg_cal_target": "catalog",
                    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "README.md").write_text(
        "# Primary matrix run outputs\n\n"
        "Baselines, inverse-power, and KL runs written here.\n"
    )
    phases = {
        "baselines": [jobs_baselines],
        "legacy": [jobs_legacy],
        "kl": [jobs_kl],
        "all": [jobs_baselines, jobs_legacy, jobs_kl],
    }
    if PHASE not in phases:
        raise SystemExit(f"Unknown PHASE={PHASE}; expected {list(phases)}")
    n = 0
    _write_status(PHASE, "starting")
    for gen in phases[PHASE]:
        for args in gen():
            _run(*args, phase=PHASE)
            n += 1
    print(f"PHASE_COMPLETE phase={PHASE} jobs_touched={n}", flush=True)
    _write_status(PHASE, f"PHASE_COMPLETE jobs_touched={n}")


if __name__ == "__main__":
    main()
