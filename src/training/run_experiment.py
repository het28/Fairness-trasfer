"""
Train/evaluate RecBole + MEG-RW. RecBole ``model`` in config selects Dataset compatibility;
``backbone`` selects the actual recommender class.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Sequence
import yaml

from logging import getLogger

import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_logger, init_seed, set_color

from training.experiment_artifacts import (
    append_manifest_row,
    save_metrics_json,
    save_resolved_config,
    setup_file_logging,
    split_type_from_config,
    utc_now_iso,
)
from training.recbole_dataset import inject_meg_rw_into_train_dataset
from meg_rw.reweight import DEFAULT_MEG_RW_FIELD
from recbole_ext import (
    WeightedBPR,
    WeightedItemKNN,
    WeightedLightGCN,
    WeightedNGCF,
    WeightedNeuMF,
)

def _parse_fracs(raw: Any, default: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4)) -> tuple[float, ...]:
    if raw is None:
        return default
    if isinstance(raw, (list, tuple)):
        vals = [float(x) for x in raw]
    else:
        vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if len(vals) < 2:
        return default
    s = sum(vals)
    if s <= 0:
        return default
    vals = [v / s for v in vals]
    return tuple(vals)


def _ensure_scipy_compat() -> None:
    """Patch SciPy API differences used by older RecBole code paths."""
    try:
        from scipy.sparse import dok_matrix  # type: ignore

        # RecBole 1.2.0 calls dok_matrix._update(...), removed in newer SciPy.
        if not hasattr(dok_matrix, "_update"):
            def _dok_update_compat(self, data_dict):
                # Newer SciPy forbids direct DOK.update(...); emulate legacy _update.
                for key, value in data_dict.items():
                    self[key] = value

            setattr(dok_matrix, "_update", _dok_update_compat)
    except Exception:
        # Non-fatal: if scipy is unavailable, RecBole will raise later.
        pass


def _ensure_torch_load_compat() -> None:
    """Keep RecBole checkpoint loading compatible with torch>=2.6 defaults."""
    try:
        original_load = torch.load

        # Avoid double-wrapping if run_experiment is called multiple times.
        if getattr(original_load, "_torch_compat_patched", False):
            return

        def _load_with_legacy_default(*args, **kwargs):
            # RecBole expects legacy behavior; checkpoints are locally generated/trusted.
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        _load_with_legacy_default._torch_compat_patched = True  # type: ignore[attr-defined]
        torch.load = _load_with_legacy_default  # type: ignore[assignment]
    except Exception:
        pass


def _canonical_recbole_model(backbone: str) -> str:
    """Map our local backbone names to RecBole model class names."""
    bb = backbone.lower().strip()
    alias = {
        "lightgcn": "LightGCN",
        "weighted_lightgcn": "LightGCN",
        "ngcf": "NGCF",
        "weighted_ngcf": "NGCF",
        "bpr": "BPR",
        "weighted_bpr": "BPR",
        "itemknn": "ItemKNN",
        "weighted_itemknn": "ItemKNN",
        "neumf": "NeuMF",
        "weighted_neumf": "NeuMF",
    }
    if bb in alias:
        return alias[bb]
    # Fallback: allow direct RecBole model names via backbone.
    return backbone


def _fd(config: Config) -> dict[str, Any]:
    return config.final_config_dict


def _dataset_from_files(config_file_list: Sequence[str]) -> str | None:
    dataset = None
    for cfg in config_file_list:
        p = Path(cfg)
        if not p.exists():
            continue
        try:
            obj = yaml.safe_load(p.read_text())
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("dataset"):
            dataset = str(obj["dataset"])
    return dataset


def _backbone_from_files(config_file_list: Sequence[str]) -> str | None:
    backbone = None
    for cfg in config_file_list:
        p = Path(cfg)
        if not p.exists():
            continue
        try:
            obj = yaml.safe_load(p.read_text())
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("backbone"):
            backbone = str(obj["backbone"])
    return backbone


def _inject_kwargs(fd: dict[str, Any], rw_fracs: tuple[float, ...]) -> dict[str, Any]:
    """Shared MEG-RW / ECIR calibration injection kwargs."""
    return dict(
        alpha=float(fd.get("meg_rw_alpha", 0.0)),
        fracs=rw_fracs,
        rw_mode=str(fd.get("meg_rw_mode", "dominance")).lower(),
        normalize_phi=bool(fd.get("meg_rw_normalize_phi", False)),
        field_name=fd.get("meg_rw_value_field") or DEFAULT_MEG_RW_FIELD,
        sem_enable=bool(fd.get("meg_sem_enable", False)),
        sem_beta=float(fd.get("meg_sem_beta", 0.0)),
        sem_dim=int(fd.get("meg_sem_dim", 128)),
        sem_clip_z=float(fd.get("meg_sem_clip_z", 2.0)),
        sem_wmin=float(fd.get("meg_sem_wmin", 0.5)),
        sem_wmax=float(fd.get("meg_sem_wmax", 2.0)),
        sem_renorm_mean_one=bool(fd.get("meg_sem_renorm_mean_one", True)),
        cal_mode=fd.get("meg_cal_mode"),
        cal_lambda_r=float(fd.get("meg_cal_lambda_r", 1.0)),
        cal_target=str(fd.get("meg_cal_target", "catalog")),
        # Default False preserves phi-only behaviour; primary runners opt in.
        multiply_c_ui=bool(fd.get("meg_rw_multiply_c_ui", False)),
        c_ui_transform=str(fd.get("meg_rw_c_ui_transform", "identity")),
        return_calibration_info=bool(fd.get("meg_rw_save_calibration_info", False)),
    )


def _build_model(config: Config, train_dataset):
    """Return ``(model, calibration_info_or_None)``."""
    fd = _fd(config)
    bb = fd["backbone"].lower()
    rw_fracs = _parse_fracs(fd.get("meg_rw_group_fracs"), default=(0.1, 0.2, 0.3, 0.4))
    if bb == "lightgcn":
        return get_model("LightGCN")(config, train_dataset), None
    if bb == "ngcf":
        return get_model("NGCF")(config, train_dataset), None

    weighted = {
        "weighted_lightgcn": WeightedLightGCN,
        "weighted_ngcf": WeightedNGCF,
        "weighted_bpr": WeightedBPR,
        "weighted_neumf": WeightedNeuMF,
        "weighted_itemknn": WeightedItemKNN,
    }
    if bb in weighted:
        kwargs = _inject_kwargs(fd, rw_fracs)
        cal_info = inject_meg_rw_into_train_dataset(train_dataset, **kwargs)
        fd["meg_rw_value_field"] = kwargs["field_name"]
        return weighted[bb](config, train_dataset), cal_info

    recbole_model = _canonical_recbole_model(bb)
    return get_model(recbole_model)(config, train_dataset), None


def _experiment_id(fd: dict[str, Any]) -> str:
    ds = str(fd.get("dataset_short") or fd["dataset"]).replace("-", "")
    model = str(fd.get("model_short") or fd.get("backbone", "model"))
    seed = int(fd["seed"])
    run_label = str(fd.get("run_label", "run"))
    return f"{ds}__{model}__seed{seed}__{run_label}"


def _alpha_cell(fd: dict[str, Any]) -> str | float:
    run_label = str(fd.get("run_label", "")).lower()
    if run_label == "baseline" or run_label.startswith("baseline__"):
        return "baseline"
    bb = str(fd.get("backbone", "")).lower()
    if "weighted" in bb:
        return float(fd.get("meg_rw_alpha", 0.0))
    return "baseline"


def _baseline_mode(fd: dict[str, Any]) -> str:
    if fd.get("fairness_baseline_eval_json"):
        return "eval_inputs"
    if fd.get("fairness_baseline_json"):
        return "report_json"
    return "none"


def run_experiment(
    config_file_list: Sequence[str] | None = None,
    config_dict: dict[str, Any] | None = None,
    saved: bool = True,
    fairness_audit: bool = False,
    fairness_report_dir: str | None = None,
    fairness_baseline_json: str | None = None,
    fairness_baseline_eval_json: str | None = None,
    fairness_save_eval_inputs: bool = False,
) -> dict:
    _ensure_scipy_compat()
    _ensure_torch_load_compat()

    config_file_list = list(config_file_list or [])
    config_dict = dict(config_dict or {})

    backbone = (
        config_dict.get("backbone")
        or _backbone_from_files(config_file_list)
        or "weighted_lightgcn"
    )
    recbole_model = _canonical_recbole_model(backbone)
    dataset = config_dict.get("dataset") or _dataset_from_files(config_file_list) or "ml-1m"

    merged_cfg = dict(config_dict)
    if fairness_report_dir is not None:
        merged_cfg["fairness_report_dir"] = fairness_report_dir
    if fairness_baseline_json is not None:
        merged_cfg["fairness_baseline_json"] = fairness_baseline_json
    if fairness_baseline_eval_json is not None:
        merged_cfg["fairness_baseline_eval_json"] = fairness_baseline_eval_json
    if fairness_save_eval_inputs:
        merged_cfg["fairness_save_eval_inputs"] = True

    config = Config(
        model=recbole_model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=merged_cfg,
    )
    fd = _fd(config)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()

    rp = Path(fd["fairness_report_dir"]) if fd.get("fairness_report_dir") else None
    manifest_path = Path(fd["manifest_path"]) if fd.get("manifest_path") else None
    started_at = utc_now_iso()
    log_fh = None
    if rp is not None:
        rp.mkdir(parents=True, exist_ok=True)
        save_resolved_config(config, rp / "config_resolved.yaml")
        log_fh = setup_file_logging(rp / "train.log")

    exp_id = _experiment_id(fd)
    status = "failed"
    test_result: Any = None
    audit: dict[str, Any] | None = None

    try:
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)

        init_seed(config["seed"] + int(fd.get("local_rank", 0)), config["reproducibility"])
        train_ds = train_data._dataset
        model, cal_info = _build_model(config, train_ds)
        model = model.to(config["device"])
        if cal_info is not None and rp is not None:
            import json as _json

            (rp / "input_calibration.json").write_text(_json.dumps(cal_info, indent=2))
            logger.info(
                "saved input_calibration.json F_in=%.6f InputGain=%.6f",
                float(cal_info.get("F_in", float("nan"))),
                float(cal_info.get("InputGain", float("nan"))),
            )
        logger.info(model)

        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        trainer.fit(
            train_data, valid_data, saved=saved, show_progress=config["show_progress"]
        )
        test_result = trainer.evaluate(
            test_data, load_best_model=saved, show_progress=config["show_progress"]
        )
        logger.info(set_color("test result", "yellow") + f": {test_result}")

        out: dict[str, Any] = {"test_result": test_result}

        if fairness_audit:
            from evaluation.fairness_audit import (
                load_fairness_report_json,
                run_fairness_audit,
                try_load_audit_result,
            )
            from evaluation.io import load_eval_inputs, save_audit_json, save_flat_audit_csv_row
            from evaluation.summarize import audit_result_from_jsonable, flatten_audit_result

            baseline_inputs = None
            bej = fd.get("fairness_baseline_eval_json")
            if bej:
                baseline_inputs = load_eval_inputs(bej)

            baseline_result = None
            baseline_audit_dict = None
            bj = fd.get("fairness_baseline_json")
            if bj and baseline_inputs is None:
                baseline_result = try_load_audit_result(bj)
                baseline_audit_dict = None if baseline_result is not None else load_fairness_report_json(bj)

            bmode = _baseline_mode(fd)
            topk_cfg = fd["topk"]
            topk_int = int(topk_cfg[0]) if isinstance(topk_cfg, (list, tuple)) else int(topk_cfg)
            audit_metadata = {
                "experiment_id": exp_id,
                "dataset": str(fd.get("dataset_short") or fd["dataset"]),
                "model": str(fd.get("model_short") or fd.get("backbone", "")),
                "trainer_backbone": str(fd.get("backbone", "")),
                "recbole_model": str(fd["model"]),
                "seed": int(fd["seed"]),
                "alpha": _alpha_cell(fd),
                "meg_rw_alpha": float(fd.get("meg_rw_alpha", 0.0)),
                "meg_sem_beta": float(fd.get("meg_sem_beta", 0.0)),
                "meg_sem_enable": bool(fd.get("meg_sem_enable", False)),
                "sota_method": str(fd.get("sota_method", "none")),
                "sota_lambda": float(fd.get("sota_lambda", 0.2)),
                "sota_candidate_mult": int(fd.get("sota_candidate_mult", 20)),
                "split_type": split_type_from_config(fd),
                "transfer_normalize": "row",
                "topk": topk_int,
                "baseline_mode": bmode,
                "fairness_baseline_eval_json": str(bej) if bej else "",
                "fairness_baseline_json": str(bj) if bj else "",
                "baseline_run_dir": str(fd.get("baseline_run_dir", "")),
                "report_dir": str(rp.resolve()) if rp else "",
                "timestamp_utc": utc_now_iso(),
            }

            eval_inputs_path = (rp / "eval_inputs.json") if rp and fd.get("fairness_save_eval_inputs") else None

            try:
                audit = run_fairness_audit(
                    config=config,
                    model=model,
                    test_data=test_data,
                    train_dataset=train_ds,
                    device=config["device"],
                    metadata=audit_metadata,
                    baseline_inputs=baseline_inputs,
                    baseline_result=baseline_result,
                    baseline_audit_dict=baseline_audit_dict,
                    eval_inputs_path=eval_inputs_path,
                    rerank_method=str(fd.get("sota_method", "none")),
                    rerank_lambda=float(fd.get("sota_lambda", 0.2)),
                    rerank_candidate_mult=int(fd.get("sota_candidate_mult", 20)),
                    item_fracs=_parse_fracs(fd.get("meg_rw_group_fracs")),
                )
            except NotImplementedError as e:
                # Some RecBole models (in certain versions) don't expose full_sort_predict.
                # Keep training/eval run successful and record audit limitation instead.
                audit = {
                    "metadata": audit_metadata,
                    "error": f"fairness_audit_skipped: {type(e).__name__}: {e}",
                }
            logger.info(set_color("fairness audit", "yellow") + f": keys={list(audit.keys())}")
            out["fairness_audit"] = audit

            if rp is not None:
                save_audit_json(audit, rp / "fairness_report.json")
                if "error" not in audit:
                    flat = flatten_audit_result(audit_result_from_jsonable(audit))
                    save_flat_audit_csv_row(flat, rp / "fairness_sweep.csv")

        status = "success"
        return out

    finally:
        finished_at = utc_now_iso()
        if rp is not None:
            metrics_payload: dict[str, Any] = {
                "experiment_id": exp_id,
                "status": status,
                "started_at": started_at,
                "finished_at": finished_at,
                "test_result": test_result,
            }
            if audit is not None:
                metrics_payload["fairness_audit"] = audit
            save_metrics_json(rp / "metrics.json", metrics_payload)

        if manifest_path is not None and rp is not None:
            append_manifest_row(
                manifest_path,
                {
                    "experiment_id": exp_id,
                    "dataset": str(fd.get("dataset_short") or fd["dataset"]),
                    "model": str(fd.get("model_short") or fd.get("backbone", "")),
                    "seed": int(fd["seed"]),
                    "alpha": _alpha_cell(fd),
                    "run_dir": str(rp.resolve()),
                    "baseline_run_dir": str(fd.get("baseline_run_dir", "")),
                    "status": status,
                    "started_at": started_at,
                    "finished_at": finished_at,
                },
            )

        if log_fh is not None:
            root = logging.getLogger()
            root.removeHandler(log_fh)
            log_fh.close()


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="MEG-RW + RecBole training")
    p.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="YAML config files (later overrides earlier)",
    )
    p.add_argument("--fairness-audit", action="store_true")
    p.add_argument(
        "--fairness-report-dir",
        default=None,
        help="If set with --fairness-audit, write fairness_report.json and append fairness_sweep.csv",
    )
    p.add_argument(
        "--fairness-baseline-json",
        default=None,
        help="Prior fairness_report.json (new schema) for ΔT; use fairness_baseline_eval_json in YAML for alignment",
    )
    p.add_argument(
        "--fairness-baseline-eval-json",
        default=None,
        help="Prior eval_inputs.json for ΔT + exposure–utility alignment",
    )
    p.add_argument(
        "--fairness-save-eval-inputs",
        action="store_true",
        help="With --fairness-report-dir, also write eval_inputs.json (large)",
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    run_experiment(
        config_file_list=args.config,
        fairness_audit=args.fairness_audit,
        fairness_report_dir=args.fairness_report_dir,
        fairness_baseline_json=args.fairness_baseline_json,
        fairness_baseline_eval_json=args.fairness_baseline_eval_json,
        fairness_save_eval_inputs=args.fairness_save_eval_inputs,
    )


if __name__ == "__main__":
    main()
