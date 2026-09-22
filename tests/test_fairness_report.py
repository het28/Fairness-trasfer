"""Fairness report shape and CSV flatten."""

from cikm_eval.report import flatten_report_for_csv


def test_flatten_report_for_csv_scalars():
    report = {
        "metadata": {"seed": 1, "dataset": "ml-1m"},
        "global": {"ndcg_mean": 0.4, "topk": 10},
        "item_side": {"mad_vs_catalog": 0.1, "exposure_share_per_group": [0.2, 0.3]},
        "user_side": {"gap_ndcg_max_min": 0.05, "per_group_ndcg_mean": [0.3, 0.5]},
        "cross_side": {"transfer_matrix_row_normalized": [[0.5, 0.5]]},
    }
    flat = flatten_report_for_csv(report)
    assert flat["meta_seed"] == 1
    assert flat["global_ndcg_mean"] == 0.4
    assert flat["item_mad_vs_catalog"] == 0.1
    assert flat["user_gap_ndcg_max_min"] == 0.05
    assert "item_exposure_share_per_group" not in flat
    assert "cross_transfer_matrix_row_normalized" not in flat
