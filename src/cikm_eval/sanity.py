"""EvalInputs validation and transfer matrix checks."""

from __future__ import annotations

import math

from cikm_eval.types import (
    ITEM_GROUP_ORDER,
    USER_GROUP_ORDER,
    EvalInputs,
    GroupName,
    TransferMatrixResult,
)


def validate_eval_inputs(inputs: EvalInputs) -> None:
    if inputs.k <= 0:
        raise ValueError("k must be positive")
    if not inputs.recommendations:
        raise ValueError("recommendations must be non-empty")
    nonempty = sum(1 for v in inputs.recommendations.values() if len(v) > 0)
    if nonempty == 0:
        raise ValueError("all recommendation lists are empty")

    for u, recs in inputs.recommendations.items():
        if u not in inputs.user_group:
            raise ValueError(f"user {u} missing from user_group")
        for it in recs[: inputs.k]:
            if it not in inputs.item_group:
                raise ValueError(f"recommended item {it} missing from item_group")


def check_transfer_matrix_rows_sum_to_one(
    result: TransferMatrixResult,
    tol: float = 1e-5,
) -> None:
    if result.normalize != "row":
        return
    for i, ug in enumerate(result.user_groups):
        s = sum(result.matrix[i])
        if not math.isfinite(s) or abs(s - 1.0) > tol:
            raise ValueError(
                f"row {ug} sums to {s}, expected 1.0 within {tol} (normalize=row)"
            )


def check_group_coverage(inputs: EvalInputs) -> dict[str, int | list[str]]:
    seen_u: set[GroupName] = set()
    seen_i: set[GroupName] = set()
    for u, recs in inputs.recommendations.items():
        seen_u.add(inputs.user_group[u])
        for it in recs[: inputs.k]:
            seen_i.add(inputs.item_group[it])

    missing_u = [g for g in USER_GROUP_ORDER if g not in seen_u]
    missing_i = [g for g in ITEM_GROUP_ORDER if g not in seen_i]
    return {
        "n_user_groups_present": len(seen_u),
        "n_item_groups_present": len(seen_i),
        "missing_user_groups": missing_u,
        "missing_item_groups": missing_i,
    }
