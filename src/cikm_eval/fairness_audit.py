"""
RecBole full-sort bridge: build :class:`EvalInputs`, run :func:`run_full_fairness_audit`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.data.interaction import Interaction

from cikm_eval.grouping import (
    item_group_labels_from_train_coo,
    normalized_popularity,
    user_mainstreamness_labels,
)
from cikm_eval.grouping_sanity import user_group_sanity_table
from cikm_eval.rerank import rerank_topk
from cikm_eval.sanity import validate_eval_inputs
from cikm_eval.summarize import audit_result_from_jsonable, audit_result_to_jsonable, run_full_fairness_audit
from cikm_eval.types import (
    EvalInputs,
    FullFairnessAuditResult,
    ItemId,
    UserId,
    item_group_name_from_index,
    user_group_name_from_index,
)


def load_fairness_report_json(path: str | Path) -> dict[str, Any]:
    """Load raw JSON (legacy dict or new audit payload)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def try_load_audit_result(path: str | Path) -> FullFairnessAuditResult | None:
    try:
        return audit_result_from_jsonable(load_fairness_report_json(path))
    except (KeyError, TypeError, ValueError):
        return None


@torch.no_grad()
def _full_sort_scores_with_predict_fallback(
    model,
    user_ids: np.ndarray,
    uid_field: str,
    iid_field: str,
    tot_item_num: int,
    device: torch.device,
    item_chunk_size: int = 4096,
) -> torch.Tensor:
    """Compute full-sort scores using model.predict(...) in item chunks.

    Used for models that do not implement full_sort_predict in current RecBole builds.
    Returns shape [batch_users, tot_item_num] on CPU.
    """
    batch_users = int(user_ids.shape[0])
    all_scores = torch.empty((batch_users, tot_item_num), dtype=torch.float32, device="cpu")
    u_t = torch.as_tensor(user_ids, dtype=torch.long, device=device)

    for start in range(0, tot_item_num, item_chunk_size):
        end = min(start + item_chunk_size, tot_item_num)
        items = torch.arange(start, end, dtype=torch.long, device=device)
        n_items = end - start

        users_rep = u_t.repeat_interleave(n_items)
        items_rep = items.repeat(batch_users)
        inter = Interaction({uid_field: users_rep, iid_field: items_rep}).to(device)
        s = model.predict(inter).view(batch_users, n_items).detach().cpu()
        all_scores[:, start:end] = s

    return all_scores


@torch.no_grad()
def build_eval_inputs_from_full_sort(
    model,
    test_data: FullSortEvalDataLoader,
    train_dataset,
    device: torch.device | str,
    topk: int = 10,
    item_fracs: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4),
    rerank_method: str = "none",
    rerank_lambda: float = 0.2,
    rerank_candidate_mult: int = 20,
) -> EvalInputs:
    """Run full-sort scoring once; return canonical :class:`EvalInputs`."""
    if not isinstance(test_data, FullSortEvalDataLoader):
        raise TypeError("test_data must be FullSortEvalDataLoader (eval_args mode=full)")

    device = torch.device(device) if isinstance(device, str) else device
    model.eval()
    if hasattr(model, "restore_user_e"):
        model.restore_user_e = None
        model.restore_item_e = None

    uid_f = train_dataset.uid_field
    iid_f = train_dataset.iid_field
    mat = train_dataset.inter_matrix(form="coo")
    n_items = train_dataset.item_num
    n_users = train_dataset.user_num
    deg, item_lab = item_group_labels_from_train_coo(
        mat.col, n_items, fracs=item_fracs
    )
    p_item = normalized_popularity(deg)
    u_train = train_dataset.inter_feat[uid_f].cpu().numpy()
    i_train = train_dataset.inter_feat[iid_f].cpu().numpy()
    user_lab = user_mainstreamness_labels(u_train, i_train, n_users, p_item)

    item_group: dict[ItemId, str] = {
        ItemId(i): item_group_name_from_index(int(item_lab[i]), n_groups=len(item_fracs))
        for i in range(n_items)
    }
    user_group: dict[UserId, str] = {
        UserId(u): user_group_name_from_index(int(user_lab[u])) for u in range(n_users)
    }

    recommendations: dict[UserId, list[ItemId]] = {}
    test_items: dict[UserId, list[ItemId]] = {}
    tot_item_num = train_dataset.item_num
    k_eff = min(topk, tot_item_num - 1)
    cand_k = int(min(tot_item_num - 1, max(k_eff, k_eff * max(1, int(rerank_candidate_mult)))))

    for batch in test_data:
        interaction, history_index, positive_u, positive_i = batch
        users = interaction[uid_f].cpu().numpy().astype(np.int64)
        try:
            scores = model.full_sort_predict(interaction.to(device))
            scores = scores.view(-1, tot_item_num)
            scores = scores.cpu()
        except (NotImplementedError, AttributeError):
            scores = _full_sort_scores_with_predict_fallback(
                model=model,
                user_ids=users,
                uid_field=uid_f,
                iid_field=iid_f,
                tot_item_num=tot_item_num,
                device=device,
            )
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        scores_np = scores.numpy()
        pu = positive_u.cpu().numpy().astype(np.int64)
        pi = positive_i.cpu().numpy().astype(np.int64)
        batch_users = scores_np.shape[0]
        for r in range(batch_users):
            u = int(users[r])
            pos_mask = pu == r
            test_items[UserId(u)] = pi[pos_mask].tolist()
            order = np.argsort(-scores_np[r], kind="mergesort")
            cand_items = order[:cand_k].astype(np.int64)
            top_items = rerank_topk(
                method=rerank_method,
                scores=scores_np[r],
                candidate_items=cand_items,
                topk=k_eff,
                item_groups=item_lab,
                item_popularity=p_item,
                lam=float(rerank_lambda),
            )
            recommendations[UserId(u)] = [ItemId(int(x)) for x in top_items]

    return EvalInputs(
        recommendations=recommendations,
        test_items=test_items,
        item_group=item_group,
        user_group=user_group,
        k=k_eff,
    )


@torch.no_grad()
def run_fairness_audit(
    config,
    model,
    test_data: FullSortEvalDataLoader,
    train_dataset,
    device: torch.device | str,
    topk: int = 10,
    metadata: Mapping[str, Any] | None = None,
    baseline_inputs: EvalInputs | None = None,
    baseline_result: FullFairnessAuditResult | None = None,
    baseline_audit_dict: Mapping[str, Any] | None = None,
    include_grouping_sanity: bool = True,
    item_fracs: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4),
    strict_validate: bool = False,
    eval_inputs_path: str | Path | None = None,
    rerank_method: str = "none",
    rerank_lambda: float = 0.2,
    rerank_candidate_mult: int = 20,
) -> dict[str, Any]:
    """
    Returns JSON-serializable dict (``audit_result_to_jsonable`` shape).

    Baseline precedence: ``baseline_inputs`` (ΔT + alignment) >
    ``baseline_result`` (ΔT only, same row/col order as saved audit) >
    ``baseline_audit_dict`` (parsed as new-format audit JSON).
    """
    del config  # reserved for future eval_args hooks

    br = baseline_result
    if br is None and baseline_audit_dict is not None:
        try:
            br = audit_result_from_jsonable(dict(baseline_audit_dict))
        except (KeyError, TypeError, ValueError):
            br = None

    try:
        inputs = build_eval_inputs_from_full_sort(
            model=model,
            test_data=test_data,
            train_dataset=train_dataset,
            device=device,
            topk=topk,
            item_fracs=item_fracs,
            rerank_method=rerank_method,
            rerank_lambda=rerank_lambda,
            rerank_candidate_mult=rerank_candidate_mult,
        )
    except TypeError as e:
        return {"error": str(e)}

    if not inputs.recommendations:
        return {"error": "no recommendations collected"}

    if eval_inputs_path is not None:
        from cikm_eval.io import save_eval_inputs

        save_eval_inputs(inputs, eval_inputs_path)

    if strict_validate:
        validate_eval_inputs(inputs)

    n_users = train_dataset.user_num
    uid_f = train_dataset.uid_field
    iid_f = train_dataset.iid_field
    mat = train_dataset.inter_matrix(form="coo")
    n_items = train_dataset.item_num
    deg, item_lab = item_group_labels_from_train_coo(mat.col, n_items, fracs=item_fracs)
    p_item = normalized_popularity(deg)
    u_train = train_dataset.inter_feat[uid_f].cpu().numpy()
    i_train = train_dataset.inter_feat[iid_f].cpu().numpy()
    user_lab = user_mainstreamness_labels(u_train, i_train, n_users, p_item)
    n_h = int(user_lab.max()) + 1

    grouping_sanity = None
    if include_grouping_sanity:
        grouping_sanity = user_group_sanity_table(
            train_dataset,
            user_lab,
            n_groups_user=n_h,
            item_fracs=item_fracs,
        )

    meta = dict(metadata) if metadata else {}
    use_result = br if baseline_inputs is None else None
    result = run_full_fairness_audit(
        inputs=inputs,
        metadata=meta,
        baseline_inputs=baseline_inputs,
        baseline_result=use_result,
        grouping_sanity=grouping_sanity,
    )
    return audit_result_to_jsonable(result)
