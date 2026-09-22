"""Canonical eval types and structured audit results (stable group ordering)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

UserId = int
ItemId = int
GroupName = str

# Global ordering: use everywhere (matrices, CSV, plots).
USER_GROUP_ORDER: tuple[GroupName, ...] = (
    "niche",
    "semi_niche",
    "semi_mainstream",
    "mainstream",
)
ITEM_GROUP_ORDER: tuple[GroupName, ...] = (
    "Head",
    "UpperMid",
    "LowerMid",
    "Tail",
)


def ordered_user_groups(seen: set[GroupName]) -> list[GroupName]:
    return [g for g in USER_GROUP_ORDER if g in seen]


def ordered_item_groups(seen: set[GroupName]) -> list[GroupName]:
    # Keep canonical order first, then append dynamic groups (e.g., G00..G19)
    out = [g for g in ITEM_GROUP_ORDER if g in seen]
    rest = [g for g in seen if g not in set(ITEM_GROUP_ORDER)]
    def _key(name: GroupName):
        s = str(name)
        if s.startswith("G"):
            try:
                return (0, int(s[1:]))
            except Exception:
                return (1, s)
        return (1, s)
    out.extend(sorted(rest, key=_key))
    return out


def user_group_name_from_index(i: int) -> GroupName:
    return USER_GROUP_ORDER[int(i)]


def item_group_name_from_index(i: int, n_groups: int | None = None) -> GroupName:
    ii = int(i)
    # Canonical Head/UpperMid/LowerMid/Tail names only for the default G=4 partition.
    if n_groups is None or int(n_groups) == len(ITEM_GROUP_ORDER):
        if 0 <= ii < len(ITEM_GROUP_ORDER):
            return ITEM_GROUP_ORDER[ii]
    return f"G{ii:02d}"


@dataclass(frozen=True)
class EvalInputs:
    """Single canonical object for all fairness metrics."""

    recommendations: Dict[UserId, List[ItemId]]
    test_items: Dict[UserId, List[ItemId]]
    item_group: Dict[ItemId, GroupName]
    user_group: Dict[UserId, GroupName]
    k: int


@dataclass
class UserGroupUtilityResult:
    group_ndcg: Dict[GroupName, float]
    group_recall: Dict[GroupName, float]
    group_size: Dict[GroupName, int]
    ndcg_gap_max_min: float
    recall_gap_max_min: float
    ndcg_std: float
    recall_std: float
    worst_group_ndcg: float
    worst_group_recall: float
    best_group_ndcg: float
    best_group_recall: float


@dataclass
class TransferMatrixResult:
    """Primary matrix is row-normalized: T[h,g] = share of h's top-k mass to item group g."""

    user_groups: List[GroupName]
    item_groups: List[GroupName]
    matrix: List[List[float]]
    row_sums: Dict[GroupName, float]
    col_sums: Dict[GroupName, float]
    normalize: str = "row"
    counts: List[List[float]] | None = None


@dataclass
class TransferDeltaResult:
    user_groups: List[GroupName]
    item_groups: List[GroupName]
    delta_matrix: List[List[float]]
    l1_shift: float
    max_abs_shift: float


@dataclass
class AlignmentResult:
    group_delta_tail_exposure: Dict[GroupName, float]
    group_delta_ndcg: Dict[GroupName, float]
    pearson_alignment: float | None
    spearman_alignment: float | None


@dataclass
class FullFairnessAuditResult:
    metadata: Dict[str, Any]
    utility_overall: Dict[str, float]
    item_fairness: Dict[str, float]
    user_fairness: UserGroupUtilityResult
    transfer: TransferMatrixResult
    transfer_delta: TransferDeltaResult | None = None
    alignment: AlignmentResult | None = None
    grouping_sanity: list[dict[str, Any]] | None = None
