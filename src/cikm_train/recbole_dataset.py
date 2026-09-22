from __future__ import annotations

import numpy as np
import torch
from recbole.data.interaction import Interaction

from meg_rw.calibration import calibrate_group_weights, weights_from_phi
from meg_rw.grouping import assign_popularity_groups
from meg_rw.reweight import DEFAULT_MEG_RW_FIELD, meg_rw_weights_for_interaction_rows
from meg_rw.semantic import semantic_weights_for_interactions


def _extract_c_ui(
    train_dataset,
    n_rows: int,
    multiply_c_ui: bool,
    c_ui_transform: str = "identity",
) -> np.ndarray:
    """Return per-row interaction strengths c_ui (default ones).

    For implicit datasets without a usable rating field, returns ones.
    LastFM listening counts are typically heavy-tailed; prefer ``log1p``.
    """
    if not multiply_c_ui:
        return np.ones(n_rows, dtype=np.float32)

    feat = train_dataset.inter_feat
    rating_field = getattr(train_dataset, "rating_field", None)
    if rating_field is None or rating_field not in feat.interaction:
        return np.ones(n_rows, dtype=np.float32)

    raw = feat[rating_field].detach().cpu().numpy().astype(np.float64)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    raw = np.maximum(raw, 0.0)

    t = (c_ui_transform or "identity").strip().lower()
    if t in {"binary", "ones", "unit"}:
        c = (raw > 0).astype(np.float64)
    elif t in {"log1p", "log"}:
        c = np.log1p(raw)
    else:
        c = raw

    # Avoid exact zeros wiping out edges that exist in the train graph.
    c = np.where(c > 0, c, 1.0)
    return c.astype(np.float32)


def inject_meg_rw_into_train_dataset(
    train_dataset,
    alpha: float,
    fracs: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4),
    rw_mode: str = "dominance",
    normalize_phi: bool = False,
    field_name: str = DEFAULT_MEG_RW_FIELD,
    sem_enable: bool = False,
    sem_beta: float = 0.0,
    sem_dim: int = 128,
    sem_clip_z: float = 2.0,
    sem_wmin: float = 0.5,
    sem_wmax: float = 2.0,
    sem_renorm_mean_one: bool = True,
    cal_mode: str | None = None,
    cal_lambda_r: float = 1.0,
    cal_target: str = "catalog",
    multiply_c_ui: bool = False,
    c_ui_transform: str = "identity",
    return_calibration_info: bool = False,
):
    """
    Add ``field_name`` to ``train_dataset.inter_feat`` (in-place).

    Final edge/sample weight:

        w = phi(g(i)) * c_ui * w_sem

    where ``c_ui`` defaults to 1 unless ``multiply_c_ui=True`` and a rating
    field is present (LastFM ``weight``). Default ``multiply_c_ui=False``
    preserves legacy CIKM behaviour.

    If ``return_calibration_info`` is True, returns a dict with train-split
    ``q_in_0``, ``q_in``, ``T``, ``F_in``, ``phi``, ``M`` (no test leakage).
    """
    from meg_rw.calibration import calibrated_mass
    from meg_rw.statistics import group_catalog_share, group_interaction_mass

    feat = train_dataset.inter_feat
    iid_f = train_dataset.iid_field
    items = feat[iid_f].detach().cpu().numpy()
    n_items = train_dataset.item_num
    uids = feat[train_dataset.uid_field].detach().cpu().numpy()
    n_rows = int(items.shape[0])
    c_ui = _extract_c_ui(
        train_dataset, n_rows=n_rows, multiply_c_ui=multiply_c_ui, c_ui_transform=c_ui_transform
    )

    cal = (cal_mode or "").strip().lower()
    ii = items.astype(np.int64, copy=False)

    # Popularity groups: interaction *counts* (stable partition; matches prior MEG-RW).
    deg_count = np.bincount(ii, minlength=n_items)
    labels = assign_popularity_groups(n_items, deg_count, fracs=fracs)
    n_groups = len(fracs)

    # Interaction mass for dominance / KL: weighted by c_ui when enabled.
    if multiply_c_ui:
        deg_mass = np.bincount(ii, weights=c_ui.astype(np.float64), minlength=n_items)
    else:
        deg_mass = deg_count.astype(np.float64)

    M = group_interaction_mass(deg_mass, labels, n_groups)
    C = group_catalog_share(labels, n_groups)
    T = C / max(float(C.sum()), 1e-12)
    q_in_0 = M / max(float(M.sum()), 1e-12)
    result = None

    if not cal:
        # Legacy path: keep historical phi construction unless mass differs.
        if multiply_c_ui:
            from meg_rw.reweight import meg_rw_phi_from_groups

            phi = meg_rw_phi_from_groups(
                deg_mass,
                labels,
                n_groups=n_groups,
                alpha=alpha,
                normalize_phi=normalize_phi,
                mode=rw_mode,
            )
            w_pop = phi[labels[ii]].astype(np.float32)
        else:
            w_pop = meg_rw_weights_for_interaction_rows(
                items,
                n_items=n_items,
                alpha=alpha,
                fracs=fracs,
                mode=rw_mode,
                normalize_phi=normalize_phi,
            )
            from meg_rw.reweight import meg_rw_phi_from_groups

            phi = meg_rw_phi_from_groups(
                deg_mass,
                labels,
                n_groups=n_groups,
                alpha=alpha,
                normalize_phi=normalize_phi,
                mode=rw_mode,
            )
    else:
        mode = cal
        if mode in {"legacy", "dominance", "meg_rw"}:
            mode = "normalized_inverse_power" if normalize_phi else "legacy_inverse_power"
        result = calibrate_group_weights(
            deg_mass,
            labels,
            n_groups=n_groups,
            mode=mode,  # type: ignore[arg-type]
            alpha=float(alpha),
            target=cal_target,
            lambda_R=float(cal_lambda_r),
        )
        phi = result.phi
        T = result.T / max(float(result.T.sum()), 1e-12)
        w_pop = weights_from_phi(ii, labels, result.phi)

    q_in = calibrated_mass(M, phi)
    F_in = float(0.5 * np.abs(q_in - T).sum())
    F_in_0 = float(0.5 * np.abs(q_in_0 - T).sum())
    InputGain = F_in_0 - F_in

    if sem_enable and abs(float(sem_beta)) > 1e-12:
        w_sem = semantic_weights_for_interactions(
            train_dataset=train_dataset,
            item_internal_ids=items,
            user_internal_ids=uids,
            beta=float(sem_beta),
            clip_z=float(sem_clip_z),
            sem_dim=int(sem_dim),
            w_min=float(sem_wmin),
            w_max=float(sem_wmax),
            renorm_mean_one=bool(sem_renorm_mean_one),
        )
    else:
        w_sem = np.ones_like(w_pop, dtype=np.float32)

    w = (w_pop * c_ui * w_sem).astype(np.float32, copy=False)
    new_d = {k: feat[k] for k in feat.interaction}
    new_d[field_name] = torch.as_tensor(w, dtype=torch.float32)
    train_dataset.inter_feat = Interaction(new_d)

    if return_calibration_info:
        return {
            "q_in_0": q_in_0.astype(float).tolist(),
            "q_in": q_in.astype(float).tolist(),
            "T": T.astype(float).tolist(),
            "M": M.astype(float).tolist(),
            "phi": np.asarray(phi, dtype=float).tolist(),
            "F_in": F_in,
            "F_in_0": F_in_0,
            "InputGain": InputGain,
            "n_train_interactions": int(n_rows),
            "n_groups": int(n_groups),
            "cal_mode": cal or ("legacy_path_alpha" if abs(float(alpha)) > 0 else "baseline"),
            "alpha": float(alpha),
            "lambda_R": float(cal_lambda_r),
            "target": str(cal_target),
        }
    return None
