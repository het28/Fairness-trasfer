"""Save/load audit JSON, CSV rows, and EvalInputs snapshots."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from cikm_eval.summarize import audit_result_to_jsonable
from cikm_eval.types import EvalInputs, FullFairnessAuditResult, ItemId, UserId


def save_audit_json(result: FullFairnessAuditResult | dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from dataclasses import is_dataclass

    payload = (
        audit_result_to_jsonable(result)
        if is_dataclass(result)
        else dict(result)
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def save_flat_audit_csv_row(flat: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat.keys()))
        if write_header:
            w.writeheader()
        w.writerow(flat)


def save_eval_inputs(inputs: EvalInputs, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = eval_inputs_to_jsonable(inputs)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_eval_inputs(path: str | Path) -> EvalInputs:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return eval_inputs_from_jsonable(d)


def eval_inputs_to_jsonable(inputs: EvalInputs) -> dict[str, Any]:
    return {
        "k": inputs.k,
        "recommendations": {str(u): list(map(int, v)) for u, v in inputs.recommendations.items()},
        "test_items": {str(u): list(map(int, v)) for u, v in inputs.test_items.items()},
        "item_group": {str(i): g for i, g in inputs.item_group.items()},
        "user_group": {str(u): g for u, g in inputs.user_group.items()},
    }


def eval_inputs_from_jsonable(d: dict[str, Any]) -> EvalInputs:
    return EvalInputs(
        k=int(d["k"]),
        recommendations={UserId(int(k)): v for k, v in d["recommendations"].items()},
        test_items={UserId(int(k)): v for k, v in d["test_items"].items()},
        item_group={ItemId(int(k)): str(v) for k, v in d["item_group"].items()},
        user_group={UserId(int(k)): str(v) for k, v in d["user_group"].items()},
    )
