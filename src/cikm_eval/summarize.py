"""Orchestrate full fairness audit and flat exports."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cikm_eval.exposure import compute_item_fairness_metrics
from cikm_eval.transfer import (
    compute_exposure_utility_alignment,
    compute_transfer_delta,
    compute_transfer_matrix,
)
from cikm_eval.types import (
    EvalInputs,
    FullFairnessAuditResult,
    UserGroupUtilityResult,
)
from cikm_eval.user_fairness import compute_per_user_utility, compute_user_group_fairness


def compute_overall_utility(inputs: EvalInputs) -> dict[str, float]:
    per_u = compute_per_user_utility(inputs)
    if not per_u:
        return {"mean_ndcg": 0.0, "mean_recall": 0.0, "num_users": 0.0, "k": float(inputs.k)}
    ndcgs = [row["ndcg"] for row in per_u.values()]
    recs = [row["recall"] for row in per_u.values()]
    return {
        "mean_ndcg": float(sum(ndcgs) / len(ndcgs)),
        "mean_recall": float(sum(recs) / len(recs)),
        "num_users": float(len(per_u)),
        "k": float(inputs.k),
    }


def run_full_fairness_audit(
    inputs: EvalInputs,
    metadata: dict[str, Any],
    item_fairness_fn: Callable[[EvalInputs], dict[str, float]] | None = None,
    baseline_inputs: EvalInputs | None = None,
    baseline_result: FullFairnessAuditResult | None = None,
    grouping_sanity: list[dict[str, Any]] | None = None,
) -> FullFairnessAuditResult:
    fn = item_fairness_fn or compute_item_fairness_metrics
    utility_overall = compute_overall_utility(inputs)
    item_fairness = fn(inputs)
    user_fairness = compute_user_group_fairness(inputs)
    transfer = compute_transfer_matrix(inputs, normalize="row")

    transfer_delta = None
    alignment = None
    if baseline_inputs is not None:
        base_t = compute_transfer_matrix(baseline_inputs, normalize="row")
        transfer_delta = compute_transfer_delta(base_t, transfer)
        alignment = compute_exposure_utility_alignment(baseline_inputs, inputs)
    elif baseline_result is not None:
        transfer_delta = compute_transfer_delta(baseline_result.transfer, transfer)

    return FullFairnessAuditResult(
        metadata=dict(metadata),
        utility_overall=utility_overall,
        item_fairness=item_fairness,
        user_fairness=user_fairness,
        transfer=transfer,
        transfer_delta=transfer_delta,
        alignment=alignment,
        grouping_sanity=grouping_sanity,
    )


def flatten_audit_result(result: FullFairnessAuditResult) -> dict[str, float | int | str]:
    """Single CSV row; expands per-group and transfer cells with stable names."""
    flat: dict[str, float | int | str] = {}
    meta = result.metadata
    for k, v in meta.items():
        flat[str(k)] = v

    uo = result.utility_overall
    flat["mean_ndcg"] = uo.get("mean_ndcg", 0.0)
    flat["mean_recall"] = uo.get("mean_recall", 0.0)
    flat["num_users"] = int(uo.get("num_users", 0))
    flat["k"] = int(uo.get("k", 0))

    it = result.item_fairness
    flat["item_exposure_deviation"] = it.get("exposure_deviation", 0.0)
    flat["item_exposure_ratio"] = it.get("exposure_ratio", 0.0)
    flat["item_tail_ratio"] = it.get("tail_ratio", 0.0)
    flat["item_tail_head_ratio"] = it.get("tail_head_ratio", 0.0)

    uf = result.user_fairness
    flat["user_ndcg_gap"] = uf.ndcg_gap_max_min
    flat["user_recall_gap"] = uf.recall_gap_max_min
    flat["user_ndcg_std"] = uf.ndcg_std
    flat["user_recall_std"] = uf.recall_std
    flat["user_worst_ndcg"] = uf.worst_group_ndcg
    flat["user_best_ndcg"] = uf.best_group_ndcg
    flat["user_worst_recall"] = uf.worst_group_recall
    flat["user_best_recall"] = uf.best_group_recall

    for g, v in uf.group_ndcg.items():
        flat[f"user_group_ndcg_{g}"] = v
    for g, v in uf.group_recall.items():
        flat[f"user_group_recall_{g}"] = v

    tr = result.transfer
    for hi, uh in enumerate(tr.user_groups):
        for gi, ig in enumerate(tr.item_groups):
            flat[f"transfer_{uh}_{ig}"] = tr.matrix[hi][gi]

    td = result.transfer_delta
    if td is not None:
        flat["transfer_l1_shift"] = td.l1_shift
        flat["transfer_max_abs_shift"] = td.max_abs_shift
        for hi, uh in enumerate(td.user_groups):
            for gi, ig in enumerate(td.item_groups):
                flat[f"transfer_delta_{uh}_{ig}"] = td.delta_matrix[hi][gi]
    else:
        flat["transfer_l1_shift"] = ""
        flat["transfer_max_abs_shift"] = ""

    al = result.alignment
    if al is not None:
        flat["align_pearson"] = al.pearson_alignment if al.pearson_alignment is not None else ""
        flat["align_spearman"] = al.spearman_alignment if al.spearman_alignment is not None else ""
        for g, v in al.group_delta_tail_exposure.items():
            flat[f"align_delta_tail_{g}"] = v
        for g, v in al.group_delta_ndcg.items():
            flat[f"align_delta_ndcg_{g}"] = v
    else:
        flat["align_pearson"] = ""
        flat["align_spearman"] = ""

    # Paper-friendly aliases (duplicate item_* scalars under shorter names)
    flat["exposure_deviation"] = flat.get("item_exposure_deviation", "")
    flat["exposure_ratio"] = flat.get("item_exposure_ratio", "")
    flat["tail_ratio"] = flat.get("item_tail_ratio", "")

    return flat


def audit_result_to_jsonable(result: FullFairnessAuditResult) -> dict[str, Any]:
    """Nested dict suitable for JSON (no dataclass objects)."""
    return _dataclass_to_dict(result)


def _dataclass_to_dict(obj: Any) -> Any:
    from dataclasses import asdict, is_dataclass

    if is_dataclass(obj):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dataclass_to_dict(x) for x in obj]
    return obj


def audit_result_from_jsonable(d: dict[str, Any]) -> FullFairnessAuditResult:
    """Load audit written by ``audit_result_to_jsonable``."""
    from cikm_eval.types import (
        AlignmentResult,
        TransferDeltaResult,
        TransferMatrixResult,
    )

    uf = d["user_fairness"]
    user_fairness = UserGroupUtilityResult(
        group_ndcg=dict(uf["group_ndcg"]),
        group_recall=dict(uf["group_recall"]),
        group_size={k: int(v) for k, v in uf["group_size"].items()},
        ndcg_gap_max_min=float(uf["ndcg_gap_max_min"]),
        recall_gap_max_min=float(uf["recall_gap_max_min"]),
        ndcg_std=float(uf["ndcg_std"]),
        recall_std=float(uf["recall_std"]),
        worst_group_ndcg=float(uf["worst_group_ndcg"]),
        worst_group_recall=float(uf["worst_group_recall"]),
        best_group_ndcg=float(uf["best_group_ndcg"]),
        best_group_recall=float(uf["best_group_recall"]),
    )

    tr = d["transfer"]
    transfer = TransferMatrixResult(
        user_groups=list(tr["user_groups"]),
        item_groups=list(tr["item_groups"]),
        matrix=tr["matrix"],
        row_sums={k: float(v) for k, v in tr["row_sums"].items()},
        col_sums={k: float(v) for k, v in tr["col_sums"].items()},
        normalize=str(tr.get("normalize", "row")),
        counts=tr.get("counts"),
    )

    td_raw = d.get("transfer_delta")
    transfer_delta = None
    if td_raw:
        transfer_delta = TransferDeltaResult(
            user_groups=list(td_raw["user_groups"]),
            item_groups=list(td_raw["item_groups"]),
            delta_matrix=td_raw["delta_matrix"],
            l1_shift=float(td_raw["l1_shift"]),
            max_abs_shift=float(td_raw["max_abs_shift"]),
        )

    al_raw = d.get("alignment")
    alignment = None
    if al_raw:
        alignment = AlignmentResult(
            group_delta_tail_exposure=dict(al_raw["group_delta_tail_exposure"]),
            group_delta_ndcg=dict(al_raw["group_delta_ndcg"]),
            pearson_alignment=_optional_float(al_raw.get("pearson_alignment")),
            spearman_alignment=_optional_float(al_raw.get("spearman_alignment")),
        )

    uo = d["utility_overall"]
    utility_overall = {
        "mean_ndcg": float(uo["mean_ndcg"]),
        "mean_recall": float(uo["mean_recall"]),
        "num_users": float(uo["num_users"]),
        "k": float(uo["k"]),
    }

    return FullFairnessAuditResult(
        metadata=dict(d["metadata"]),
        utility_overall=utility_overall,
        item_fairness={k: float(v) for k, v in d["item_fairness"].items()},
        user_fairness=user_fairness,
        transfer=transfer,
        transfer_delta=transfer_delta,
        alignment=alignment,
        grouping_sanity=d.get("grouping_sanity"),
    )


def _optional_float(x: Any) -> float | None:
    if x is None or x == "":
        return None
    v = float(x)
    return v if __import__("math").isfinite(v) else None
