from __future__ import annotations

import numpy as np

from meg_rw.calibration import (
    calibrate_group_weights,
    calibrated_mass,
    kl_divergence,
    phi_exponential,
    phi_legacy_inverse_power,
    phi_kl_projection,
)
from meg_rw.grouping import assign_popularity_groups


def _toy():
    # 100 items, degrees heavily head-skewed
    deg = np.zeros(100, dtype=np.int64)
    deg[:10] = 50
    deg[10:30] = 10
    deg[30:60] = 3
    deg[60:] = 1
    labels = assign_popularity_groups(100, deg, fracs=(0.1, 0.2, 0.3, 0.4))
    return deg, labels


def test_legacy_alpha0_is_ones():
    deg, lab = _toy()
    r = calibrate_group_weights(deg, lab, 4, mode="legacy_inverse_power", alpha=0.0)
    assert np.allclose(r.phi, 1.0)


def test_exponential_matches_inverse_power():
    D = np.array([2.0, 1.0, 0.5, 0.25])
    a = 0.4
    assert np.allclose(phi_legacy_inverse_power(D, a), phi_exponential(D, a), rtol=1e-10)


def test_kl_projection_moves_mass_toward_catalog():
    deg, lab = _toy()
    r0 = calibrate_group_weights(deg, lab, 4, mode="legacy_inverse_power", alpha=0.0)
    r = calibrate_group_weights(
        deg, lab, 4, mode="kl_projection", target="catalog", lambda_R=0.1
    )
    q0 = calibrated_mass(r0.M, np.ones_like(r0.M))
    q1 = calibrated_mass(r.M, np.exp(r.extras["u"]))
    assert kl_divergence(q1, r.T) < kl_divergence(q0, r.T)


def test_kl_projection_positive_finite():
    M = np.array([0.7, 0.2, 0.07, 0.03])
    T = np.array([0.1, 0.2, 0.3, 0.4])
    phi, extras = phi_kl_projection(M, T, lambda_R=1.0)
    assert np.all(np.isfinite(phi))
    assert np.all(phi > 0)


def test_legacy_not_equal_kl_optimum_in_general():
    deg, lab = _toy()
    leg = calibrate_group_weights(deg, lab, 4, mode="legacy_inverse_power", alpha=0.4)
    kl = calibrate_group_weights(deg, lab, 4, mode="kl_projection", lambda_R=0.5)
    # They may correlate but should not be identical in general.
    corr = np.corrcoef(leg.phi, kl.phi)[0, 1]
    assert np.isfinite(corr)
    assert not np.allclose(leg.phi, kl.phi, rtol=1e-3)
