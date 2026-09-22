"""Structured fairness reports: JSON + flat CSV for sweeps."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping


def save_fairness_report(
    report: Mapping[str, Any],
    json_path: str | Path,
    csv_path: str | Path | None = None,
) -> None:
    """Write full nested ``report`` as JSON; optionally one CSV row (flattened scalars)."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    if csv_path is None:
        return
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    flat = flatten_report_for_csv(report)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat.keys()))
        if write_header:
            w.writeheader()
        w.writerow(flat)


def flatten_report_for_csv(report: Mapping[str, Any]) -> dict[str, Any]:
    """Single-row flatten for sweep aggregation (no nested lists)."""
    if isinstance(report, dict) and "user_fairness" in report and "utility_overall" in report:
        from cikm_eval.summarize import audit_result_from_jsonable, flatten_audit_result

        return flatten_audit_result(audit_result_from_jsonable(report))

    meta = report.get("metadata", {})
    g = report.get("global", {})
    it = report.get("item_side", {})
    us = report.get("user_side", {})
    cs = report.get("cross_side", {})
    out: dict[str, Any] = {}
    for k, v in meta.items():
        out[f"meta_{k}"] = v
    for k, v in g.items():
        out[f"global_{k}"] = v
    for k, v in it.items():
        if isinstance(v, (list, dict)):
            continue
        out[f"item_{k}"] = v
    for k, v in us.items():
        if isinstance(v, (list, dict)):
            continue
        out[f"user_{k}"] = v
    for k, v in cs.items():
        if isinstance(v, (list, dict)):
            continue
        out[f"cross_{k}"] = v
    return out
