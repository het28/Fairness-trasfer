from __future__ import annotations

import numpy as np


def _as_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim > 2:
        return x.reshape(x.shape[0], -1)
    return x


def _item_feature_matrix(train_dataset, max_raw_dim: int = 512) -> np.ndarray:
    """Build dense item feature matrix from RecBole item_feat fields.

    Falls back to empty matrix when no usable item-side features exist.
    """
    item_feat = getattr(train_dataset, "item_feat", None)
    iid_field = getattr(train_dataset, "iid_field", None)
    if item_feat is None or not hasattr(item_feat, "interaction"):
        return np.empty((train_dataset.item_num, 0), dtype=np.float32)

    cols: list[np.ndarray] = []
    for field in item_feat.interaction:
        if field == iid_field:
            continue
        v = item_feat[field].detach().cpu().numpy()
        v = _as_2d(v).astype(np.float32, copy=False)
        if v.shape[1] > max_raw_dim:
            v = v[:, :max_raw_dim]
        # Scale each column to avoid huge raw ID magnitudes dominating.
        denom = np.maximum(np.nanpercentile(np.abs(v), 99, axis=0), 1.0)
        v = v / denom
        cols.append(v)

    if not cols:
        return np.empty((train_dataset.item_num, 0), dtype=np.float32)
    x = np.concatenate(cols, axis=1).astype(np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def _reduce_item_semantics(x: np.ndarray, sem_dim: int = 128) -> np.ndarray:
    """Project item features to compact semantic vectors with SVD."""
    n_items, d = x.shape
    if d == 0:
        return np.zeros((n_items, 1), dtype=np.float32)
    if d <= sem_dim:
        return x.astype(np.float32, copy=False)

    # Center then project by right singular vectors.
    xc = x - x.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
        proj = vt[:sem_dim].T
        z = xc @ proj
    except np.linalg.LinAlgError:
        z = xc[:, :sem_dim]
    return z.astype(np.float32, copy=False)


def semantic_weights_for_interactions(
    train_dataset,
    item_internal_ids: np.ndarray,
    user_internal_ids: np.ndarray,
    beta: float,
    clip_z: float = 2.0,
    sem_dim: int = 128,
    w_min: float = 0.5,
    w_max: float = 2.0,
    renorm_mean_one: bool = True,
) -> np.ndarray:
    """Compute semantic factors per train interaction row.

    Returns ones when beta==0 or semantic features are unavailable.
    """
    if abs(float(beta)) < 1e-12:
        return np.ones_like(item_internal_ids, dtype=np.float32)

    item_x = _item_feature_matrix(train_dataset)
    item_z = _reduce_item_semantics(item_x, sem_dim=sem_dim)

    if item_z.shape[1] == 0:
        return np.ones_like(item_internal_ids, dtype=np.float32)

    # Build user semantic profile as mean of consumed item vectors.
    n_users = int(train_dataset.user_num)
    d = int(item_z.shape[1])
    u_sum = np.zeros((n_users, d), dtype=np.float32)
    u_cnt = np.bincount(user_internal_ids.astype(np.int64), minlength=n_users).astype(np.float32)
    np.add.at(u_sum, user_internal_ids.astype(np.int64), item_z[item_internal_ids.astype(np.int64)])
    u_vec = u_sum / np.maximum(u_cnt[:, None], 1.0)

    ii = item_internal_ids.astype(np.int64)
    uu = user_internal_ids.astype(np.int64)
    iv = item_z[ii]
    uv = u_vec[uu]

    iv_n = np.linalg.norm(iv, axis=1)
    uv_n = np.linalg.norm(uv, axis=1)
    sim = (iv * uv).sum(axis=1) / np.maximum(iv_n * uv_n, 1e-8)
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    mu = float(sim.mean())
    std = float(sim.std())
    z = (sim - mu) / max(std, 1e-6)
    z = np.clip(z, -clip_z, clip_z)

    w = np.exp(float(beta) * z).astype(np.float32)
    if renorm_mean_one:
        w = w / max(float(w.mean()), 1e-6)
    w = np.clip(w, w_min, w_max)
    return w.astype(np.float32, copy=False)


def semantic_profiles_from_train_dataset(
    train_dataset,
    sem_dim: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (item_semantic_vectors, user_semantic_profiles).

    Shapes:
      - item_semantic_vectors: [n_items, sem_dim_eff]
      - user_semantic_profiles: [n_users, sem_dim_eff]
    """
    item_x = _item_feature_matrix(train_dataset)
    item_z = _reduce_item_semantics(item_x, sem_dim=sem_dim)
    n_users = int(train_dataset.user_num)
    d = int(item_z.shape[1])
    if d == 0:
        return item_z, np.zeros((n_users, 1), dtype=np.float32)

    feat = train_dataset.inter_feat
    uids = feat[train_dataset.uid_field].detach().cpu().numpy().astype(np.int64)
    iids = feat[train_dataset.iid_field].detach().cpu().numpy().astype(np.int64)

    u_sum = np.zeros((n_users, d), dtype=np.float32)
    u_cnt = np.bincount(uids, minlength=n_users).astype(np.float32)
    np.add.at(u_sum, uids, item_z[iids])
    u_vec = u_sum / np.maximum(u_cnt[:, None], 1.0)
    return item_z.astype(np.float32, copy=False), u_vec.astype(np.float32, copy=False)
