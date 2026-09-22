"""Flat CSV export from FullFairnessAuditResult."""

from cikm_eval.summarize import audit_result_from_jsonable, audit_result_to_jsonable, flatten_audit_result
from cikm_eval.types import (
    FullFairnessAuditResult,
    TransferMatrixResult,
    UserGroupUtilityResult,
)


def _minimal_result(
    *,
    with_delta: bool = False,
    with_align: bool = False,
) -> FullFairnessAuditResult:
    from cikm_eval.types import AlignmentResult, TransferDeltaResult

    uf = UserGroupUtilityResult(
        group_ndcg={"niche": 0.1, "mainstream": 0.2},
        group_recall={"niche": 0.05, "mainstream": 0.06},
        group_size={"niche": 10, "mainstream": 20},
        ndcg_gap_max_min=0.1,
        recall_gap_max_min=0.01,
        ndcg_std=0.05,
        recall_std=0.005,
        worst_group_ndcg=0.1,
        worst_group_recall=0.05,
        best_group_ndcg=0.2,
        best_group_recall=0.06,
    )
    tr = TransferMatrixResult(
        user_groups=["niche", "mainstream"],
        item_groups=["Head", "Tail"],
        matrix=[[0.7, 0.3], [0.8, 0.2]],
        row_sums={"niche": 1.0, "mainstream": 1.0},
        col_sums={"Head": 1.5, "Tail": 0.5},
        normalize="row",
    )
    td = None
    if with_delta:
        td = TransferDeltaResult(
            user_groups=["niche", "mainstream"],
            item_groups=["Head", "Tail"],
            delta_matrix=[[0.1, -0.1], [0.0, 0.0]],
            l1_shift=0.2,
            max_abs_shift=0.1,
        )
    al = None
    if with_align:
        al = AlignmentResult(
            group_delta_tail_exposure={"niche": 0.1, "mainstream": -0.05},
            group_delta_ndcg={"niche": 0.02, "mainstream": -0.01},
            pearson_alignment=0.5,
            spearman_alignment=0.5,
        )
    return FullFairnessAuditResult(
        metadata={"dataset": "x", "seed": 0},
        utility_overall={"mean_ndcg": 0.15, "mean_recall": 0.055, "num_users": 30.0, "k": 10.0},
        item_fairness={"exposure_deviation": 0.1, "exposure_ratio": 0.3, "tail_ratio": 0.2, "tail_head_ratio": 1.0},
        user_fairness=uf,
        transfer=tr,
        transfer_delta=td,
        alignment=al,
        grouping_sanity=None,
    )


def test_flatten_no_optional_blocks():
    r = _minimal_result()
    flat = flatten_audit_result(r)
    assert flat["dataset"] == "x"
    assert flat["mean_ndcg"] == 0.15
    assert flat["user_ndcg_gap"] == 0.1
    assert flat["transfer_niche_Head"] == 0.7
    assert flat["transfer_l1_shift"] == ""


def test_flatten_round_trip_jsonable():
    r = _minimal_result(with_delta=True, with_align=True)
    d = audit_result_to_jsonable(r)
    r2 = audit_result_from_jsonable(d)
    flat = flatten_audit_result(r2)
    assert flat["transfer_l1_shift"] == 0.2
    assert flat["align_pearson"] == 0.5
