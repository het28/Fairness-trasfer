"""Modular group-calibration family for ECIR 2027.

Preserves legacy MEG-RW as mode ``legacy_inverse_power``.
Does NOT change default behavior of ``meg_rw.reweight`` call sites.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from meg_rw.statistics import dominance, group_catalog_share, group_interaction_mass

CorrectionMode = Literal[
    "legacy_inverse_power",
    "normalized_inverse_power",
    "exponential",
    "softmax",
    "kl_projection",
]


@dataclass(frozen=True)
class CalibrationResult:
    phi: np.ndarray
    mode: str
    D: np.ndarray
    C: np.ndarray
    M: np.ndarray
    T: np.ndarray
    extras: dict


def _target_from_name(C: np.ndarray, name: str) -> np.ndarray:
    name = name.lower()
    g = C.shape[0]
    if name in {"catalog", "catalog_share"}:
        t = C.copy()
    elif name in {"uniform", "equal"}:
        t = np.ones(g, dtype=np.float64) / g
    elif name in {"mass", "historical", "interaction"}:
        # control target: no redistribution pressure beyond mass itself
        t = np.ones(g, dtype=np.float64) / g  # caller should pass M explicitly if needed
    else:
        raise ValueError(f"unknown target {name}")
    t = np.clip(t, 1e-12, None)
    return t / t.sum()


def compute_group_stats(
    degrees: np.ndarray,
    labels: np.ndarray,
    n_groups: int,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    C = group_catalog_share(labels, n_groups)
    M = group_interaction_mass(degrees, labels, n_groups)
    D = dominance(M, C, eps)
    return C, M, D


def phi_legacy_inverse_power(D: np.ndarray, alpha: float, eps: float = 1e-8) -> np.ndarray:
    """Legacy MEG-RW: phi_g = (D_g + eps)^(-alpha)."""
    return np.power(D + eps, -float(alpha), dtype=np.float64)


def phi_normalized_inverse_power(D: np.ndarray, alpha: float, eps: float = 1e-8) -> np.ndarray:
    phi = phi_legacy_inverse_power(D, alpha, eps)
    s = phi.sum()
    if s <= 0:
        return np.ones_like(phi)
    return phi * (phi.size / s)


def phi_exponential(D: np.ndarray, alpha: float, eps: float = 1e-8) -> np.ndarray:
    """Log-domain / exponential correction: phi_g = exp(-alpha * log(D_g+eps)).

    Algebraically identical to inverse-power for positive D, included as an
    explicit numerically-stable parameterization for ablations.
    """
    return np.exp(-float(alpha) * np.log(D + eps))


def phi_softmax(D: np.ndarray, alpha: float, eps: float = 1e-8) -> np.ndarray:
    """Softmax-normalized correction over -alpha * log(D+eps).

    Produces a simplex-valued group factor (sums to 1). Downstream weighting
    should treat relative ratios; we rescale to mean 1 for backbone stability.
    """
    logits = -float(alpha) * np.log(D + eps)
    logits = logits - logits.max()
    e = np.exp(logits)
    p = e / e.sum()
    return p * p.size  # mean 1


def calibrated_mass(M: np.ndarray, w: np.ndarray) -> np.ndarray:
    q = M * w
    s = q.sum()
    if s <= 0:
        return np.ones_like(M) / M.size
    return q / s


def kl_divergence(q: np.ndarray, t: np.ndarray, eps: float = 1e-12) -> float:
    q = np.clip(q, eps, None)
    t = np.clip(t, eps, None)
    q = q / q.sum()
    t = t / t.sum()
    return float(np.sum(q * np.log(q / t)))


def phi_kl_projection(
    M: np.ndarray,
    T: np.ndarray,
    lambda_R: float = 1.0,
    max_iter: int = 500,
    lr: float = 0.05,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """Minimum-perturbation KL calibration in log-weight space.

    Optimize u_g = log w_g unconstrained:

        min_u  KL(q(w) || T) + lambda_R * sum_g M_g * u_g^2

    where q_g = M_g w_g / sum_h M_h w_h, w=exp(u).

    Regularizer rationale:
      - penalizes log-weight deviation from 0 (i.e., w=1), weighted by mass M_g,
        so popular groups are not moved "for free";
      - keeps w>0 automatically;
      - is smooth and strongly convex in u for lambda_R>0 when combined with KL.

    This is NOT claimed equivalent to legacy inverse-power.
    """
    M = np.asarray(M, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    T = np.clip(T, eps, None)
    T = T / T.sum()
    u = np.zeros_like(M)
    history = []
    for it in range(max_iter):
        w = np.exp(u)
        q = calibrated_mass(M, w)
        # dKL/dq_g = log(q_g/T_g) + 1; through q(w)
        # Use autodiff-free gradient via quotient rule.
        # Let s = sum M_h w_h, q_g = M_g w_g / s
        s = float((M * w).sum())
        # Softmax-like Jacobian for mass reweighting
        # dq_g/du_k = q_g*(1[g=k] - q_k)   because w=exp(u), q is softmax of (log M + u)
        # Actually q_g ∝ M_g exp(u_g), so yes q is softmax(log M + u).
        log_ratio = np.log(np.clip(q, eps, None) / T)
        # grad_u KL = (I - 1 q^T) log_ratio  elementwise for softmax parameterization
        # dKL/du = q * (log_ratio - <q, log_ratio>)
        centered = log_ratio - float(np.dot(q, log_ratio))
        grad_kl = q * centered
        grad_reg = 2.0 * float(lambda_R) * M * u
        grad = grad_kl + grad_reg
        u = u - lr * grad
        loss = kl_divergence(q, T) + float(lambda_R) * float(np.sum(M * u * u))
        history.append(loss)
        if it > 10 and abs(history[-1] - history[-2]) < 1e-12:
            break
    w = np.exp(u)
    # Rescale to mean 1 for comparable magnitude with legacy phi
    phi = w * (w.size / max(w.sum(), eps))
    extras = {
        "u": u,
        "w_raw": w,
        "q": calibrated_mass(M, w),
        "loss_path": history,
        "final_kl": kl_divergence(calibrated_mass(M, w), T),
        "lambda_R": lambda_R,
    }
    return phi, extras


def calibrate_group_weights(
    degrees: np.ndarray,
    labels: np.ndarray,
    n_groups: int,
    mode: CorrectionMode = "legacy_inverse_power",
    alpha: float = 0.0,
    eps: float = 1e-8,
    target: str | np.ndarray = "catalog",
    lambda_R: float = 1.0,
) -> CalibrationResult:
    C, M, D = compute_group_stats(degrees, labels, n_groups, eps=eps)
    if isinstance(target, str):
        if target.lower() in {"mass", "historical", "interaction"}:
            T = M / max(M.sum(), eps)
        else:
            T = _target_from_name(C, target)
    else:
        T = np.asarray(target, dtype=np.float64)
        T = T / T.sum()

    extras: dict = {}
    if mode == "legacy_inverse_power":
        phi = phi_legacy_inverse_power(D, alpha, eps)
    elif mode == "normalized_inverse_power":
        phi = phi_normalized_inverse_power(D, alpha, eps)
    elif mode == "exponential":
        phi = phi_exponential(D, alpha, eps)
    elif mode == "softmax":
        phi = phi_softmax(D, alpha, eps)
    elif mode == "kl_projection":
        phi, extras = phi_kl_projection(M, T, lambda_R=lambda_R, eps=eps)
    else:
        raise ValueError(f"unknown mode {mode}")

    return CalibrationResult(phi=phi.astype(np.float64), mode=mode, D=D, C=C, M=M, T=T, extras=extras)


def weights_from_phi(item_ids: np.ndarray, labels: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return phi[labels[item_ids.astype(np.int64)]].astype(np.float32)
