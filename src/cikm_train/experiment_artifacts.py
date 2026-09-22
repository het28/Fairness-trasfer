"""Resolved config, train.log, metrics.json, and sweep manifest rows."""

from __future__ import annotations

import copy
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

MANIFEST_FIELDS = [
    "experiment_id",
    "dataset",
    "model",
    "seed",
    "alpha",
    "run_dir",
    "baseline_run_dir",
    "status",
    "started_at",
    "finished_at",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def split_type_from_config(config: Mapping[str, Any]) -> str:
    ev = config.get("eval_args") or {}
    parts = [
        f"split={ev.get('split', '?')}",
        f"order={ev.get('order', '?')}",
        f"group_by={ev.get('group_by', '?')}",
        f"mode={ev.get('mode', '?')}",
    ]
    return ";".join(parts)


def save_resolved_config(config: Any, path: Path) -> None:
    """Write RecBole ``final_config_dict`` as YAML (best-effort serialization)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    d = copy.deepcopy(getattr(config, "final_config_dict", {}))

    def _norm(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): _norm(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_norm(x) for x in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_norm(d), f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def setup_file_logging(log_path: Path, level: int = logging.INFO) -> logging.FileHandler:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh.setFormatter(fmt)
    root = logging.getLogger()
    root.addHandler(fh)
    for name in ("recbole", "cikm_train"):
        logging.getLogger(name).setLevel(level)
    return fh


def save_metrics_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, default=str)


def append_manifest_row(manifest_path: Path, row: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest_path.exists()
    out_row = {k: row.get(k, "") for k in MANIFEST_FIELDS}
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(out_row)
