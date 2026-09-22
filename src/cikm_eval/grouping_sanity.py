"""Dataset-level sanity checks for user mainstreamness groups (paper tables)."""

from __future__ import annotations

from typing import Any

import numpy as np

from cikm_eval.grouping import (
    item_group_labels_from_train_coo,
    normalized_popularity,
)


def user_group_sanity_table(
    train_dataset,
    user_lab: np.ndarray,
    n_groups_user: int = 4,
    item_fracs: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4),
) -> list[dict[str, Any]]:
    """
    One row per user group ``h``: size, mean consumed item degree, mean normalized
    popularity of consumed items, mean tail-share in **train** history.
    """
    uid_f = train_dataset.uid_field
    iid_f = train_dataset.iid_field
    mat = train_dataset.inter_matrix(form="coo")
    n_items = train_dataset.item_num
    deg, item_lab = item_group_labels_from_train_coo(mat.col, n_items, fracs=item_fracs)
    tail_g = int(item_lab.max())
    p_item = normalized_popularity(deg)

    u = train_dataset.inter_feat[uid_f].cpu().numpy().astype(np.int64)
    i = train_dataset.inter_feat[iid_f].cpu().numpy().astype(np.int64)

    rows: list[dict[str, Any]] = []
    for h in range(n_groups_user):
        user_ids = np.where(user_lab == h)[0]
        size = int(user_ids.size)
        mean_deg_list: list[float] = []
        mean_p_list: list[float] = []
        tail_list: list[float] = []
        for uu in user_ids:
            m = u == uu
            if not np.any(m):
                continue
            items_u = i[m]
            mean_deg_list.append(float(deg[items_u].mean()))
            mean_p_list.append(float(p_item[items_u].mean()))
            tail_list.append(float(np.mean(item_lab[items_u] == tail_g)))

        rows.append(
            {
                "user_group": h,
                "group_size": size,
                "mean_consumed_item_degree": float(np.mean(mean_deg_list))
                if mean_deg_list
                else None,
                "mean_normalized_popularity_of_consumed": float(np.mean(mean_p_list))
                if mean_p_list
                else None,
                "mean_tail_share_in_train_history": float(np.mean(tail_list))
                if tail_list
                else None,
            }
        )
    return rows
