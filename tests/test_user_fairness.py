"""User-group NDCG / Recall aggregation (macro means)."""

from cikm_eval.types import EvalInputs
from cikm_eval.user_fairness import (
    aggregate_user_group_utility,
    compute_per_user_utility,
    compute_user_group_fairness,
    ndcg_at_k,
    recall_at_k,
)


def test_ndcg_recall_helpers():
    rel = {1, 2}
    assert ndcg_at_k([3, 1, 4], rel, k=3) > 0.0
    assert recall_at_k([3, 1, 4], rel, k=3) == 1.0 / 2.0
    assert ndcg_at_k([1], rel, k=10) < 1.0  # IDCG uses two positives; one hit → < 1
    assert ndcg_at_k([1, 2], rel, k=10) == 1.0
    assert ndcg_at_k([], rel, k=10) == 0.0
    assert recall_at_k([1], set(), k=10) == 0.0


def test_compute_user_group_fairness_toy():
    # 4 users, 2 user groups; k=2; binary relevance
    inputs = EvalInputs(
        recommendations={
            0: [10, 11],
            1: [10, 12],
            2: [11, 12],
            3: [10, 11],
        },
        test_items={
            0: [10],
            1: [12],
            2: [11],
            3: [12],
        },
        item_group={10: "Head", 11: "Tail", 12: "Head"},
        user_group={0: "niche", 1: "niche", 2: "mainstream", 3: "mainstream"},
        k=2,
    )
    per_u = compute_per_user_utility(inputs)
    assert per_u[0]["user_group"] == "niche"
    uf = aggregate_user_group_utility(per_u)
    # niche: users 0,1 — NDCG values from ranks
    g_niche_ndcg = (per_u[0]["ndcg"] + per_u[1]["ndcg"]) / 2
    g_main_ndcg = (per_u[2]["ndcg"] + per_u[3]["ndcg"]) / 2
    assert abs(uf.group_ndcg["niche"] - g_niche_ndcg) < 1e-9
    assert abs(uf.group_ndcg["mainstream"] - g_main_ndcg) < 1e-9
    assert uf.group_size["niche"] == 2
    assert uf.group_size["mainstream"] == 2
    assert uf.ndcg_gap_max_min == max(uf.group_ndcg.values()) - min(uf.group_ndcg.values())
    assert uf.worst_group_ndcg == min(uf.group_ndcg.values())
    assert uf.best_group_ndcg == max(uf.group_ndcg.values())
    one_shot = compute_user_group_fairness(inputs)
    assert one_shot.ndcg_gap_max_min == uf.ndcg_gap_max_min
