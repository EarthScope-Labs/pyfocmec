"""Fault-geometry conversions used by FOCMEC (ports of fltsol, an2dsr, v2trpl ...).

Conventions follow the original Fortran: X = north, Y = east, Z = down;
dip/strike/rake follow Aki & Richards; ``A`` is the slip vector and ``N`` the
fault normal.  All angle arguments/returns are in **radians** unless a function
name says otherwise.  Functions accept scalars or arrays and broadcast.
"""

from __future__ import annotations

import numpy as np

_TOL = 1.0e-4


def snap(x):
    """Snap near-zero components to 0 and near-unit components to +/-1.

    Mirrors the clean-up done in ``oksol.f`` and ``v2trpl.f`` (tolerance 1e-4).
    """
    x = np.array(x, copy=True)
    x = np.where(np.abs(x) < _TOL, 0.0, x)
    near1 = np.abs(np.abs(x) - 1.0) < _TOL
    return np.where(near1, np.sign(x), x)


def fault_vectors(plunge, trend, angle):
    """Port of ``fltsol.f``: A and N vectors from B-axis trend/plunge and Angle.

    Parameters are arrays (radians) of equal length ``m``.  Returns ``(A, N)``,
    each shaped ``(m, 3)``.
    """
    plunge = np.asarray(plunge)
    trend = np.asarray(trend)
    angle = np.asarray(angle)
    ct, st = np.cos(trend), np.sin(trend)
    cp, sp = np.cos(plunge), np.sin(plunge)
    # BMATRX = YROT @ ZROT, with
    #   ZROT = [[ct, st, 0], [-st, ct, 0], [0, 0, 1]]
    #   YROT = [[sp, 0, -cp], [0, 1, 0], [cp, 0, sp]]
    zero = np.zeros_like(ct)
    one = np.ones_like(ct)
    z = np.stack([np.stack([ct, st, zero], -1), np.stack([-st, ct, zero], -1), np.stack([zero, zero, one], -1)], -2)
    y = np.stack([np.stack([sp, zero, -cp], -1), np.stack([zero, one, zero], -1), np.stack([cp, zero, sp], -1)], -2)
    b = y @ z  # (m,3,3)
    ca, sa = np.cos(angle), np.sin(angle)
    # ANB = ZROT(angle) @ BMATRX; A = -ANB[1,:]; N = ANB[0,:]
    row0 = ca[..., None] * b[..., 0, :] + sa[..., None] * b[..., 1, :]
    row1 = -sa[..., None] * b[..., 0, :] + ca[..., None] * b[..., 1, :]
    return -row1, row0


def dsr_to_an(dip, strike, rake):
    """A and N vectors from dip/strike/rake in radians (as in ``lratio.f``)."""
    dip, strike, rake = (np.asarray(v) for v in (dip, strike, rake))
    a = np.stack(
        [
            np.cos(rake) * np.cos(strike) + np.sin(rake) * np.cos(dip) * np.sin(strike),
            np.cos(rake) * np.sin(strike) - np.sin(rake) * np.cos(dip) * np.cos(strike),
            -np.sin(rake) * np.sin(dip),
        ],
        -1,
    )
    n = np.stack(
        [-np.sin(strike) * np.sin(dip), np.cos(strike) * np.sin(dip), -np.cos(dip)],
        -1,
    )
    return a, n


def an_to_dsr(a, n):
    """Port of ``an2dsr.f``: dip, strike, rake (radians) from A and N.

    ``a`` and ``n`` have shape ``(..., 3)``; returns an array ``(..., 3)``.
    """
    a = np.asarray(a)
    n = np.asarray(n)
    pi = np.pi
    with np.errstate(all="ignore"):
        n3m1 = n[..., 2] == -1.0
        s2 = np.where(n3m1, np.arctan2(a[..., 1], a[..., 0]), np.arctan2(-n[..., 0], n[..., 1]))
        sin2, cos2 = np.sin(s2), np.cos(s2)
        d_a = np.arctan2(-n[..., 0] / sin2, -n[..., 2])
        d_b = np.arctan2(n[..., 1] / cos2, -n[..., 2])
        d1 = np.where(np.abs(sin2) >= 0.1, d_a, d_b)
        d1 = np.where(n[..., 2] == 0.0, 0.5 * pi, d1)
        d1 = np.where(n3m1, 0.0, d1)

        a1 = a[..., 0] * np.cos(s2) + a[..., 1] * np.sin(s2)
        a1 = np.where(np.abs(a1) < _TOL, 0.0, a1)
        # branch: a(3) != 0
        r_nz = np.where(
            d1 != 0.0,
            np.arctan2(-a[..., 2] / np.sin(d1), a1),
            np.arctan2(-1000000.0 * a[..., 2], a1),
        )
        # branch: a(3) == 0
        a2 = a[..., 0] * np.sin(s2) - a[..., 1] * np.cos(s2)
        a2 = np.where(np.abs(a2) < _TOL, 0.0, a2)
        s22 = np.sin(2 * s2)
        r_z1 = np.arctan2(a2 / s22, a1)
        r_z2 = np.arccos(np.clip(a[..., 1] / np.sin(s2), -1.0, 1.0))
        r_z3 = np.arccos(np.clip(a1, -1.0, 1.0))  # an2dsr.f takes the unclamped acos(a1) and can return NaN
        r_z = np.where(np.abs(s22) >= _TOL, r_z1, np.where(np.abs(np.sin(s2)) >= _TOL, r_z2, r_z3))
        rk = np.where(a[..., 2] != 0.0, r_nz, r_z)

        neg = d1 < 0.0
        d1 = np.where(neg, d1 + pi, d1)
        rk = np.where(neg, pi - rk, rk)
        rk = np.where(neg & (rk > pi), rk - 2 * pi, rk)
        big = d1 > 0.5 * pi
        d1 = np.where(big, pi - d1, d1)
        s2 = np.where(big, s2 + pi, s2)
        rk = np.where(big, -rk, rk)
        s2 = np.where(big & (s2 >= 2 * pi), s2 - 2 * pi, s2)
        s2 = np.where(s2 < 0.0, s2 + 2 * pi, s2)
    return np.stack([d1, s2, rk], -1)


def vector_to_trend_plunge(v):
    """Port of ``v2trpl.f``: trend and plunge (radians) of a lower-hemisphere vector.

    ``v`` has shape ``(..., 3)``; returns ``(..., 2)``.  Trend is clockwise from
    north in [0, 2pi); plunge is in [0, pi/2].  Upward vectors are reversed.
    """
    v = snap(np.asarray(v))
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    pi = np.pi
    with np.errstate(all="ignore"):
        vert = np.abs(z) == 1.0
        t = np.where(
            np.abs(x) < _TOL,
            np.where(y > 0.0, 0.5 * pi, np.where(y < 0.0, 1.5 * pi, 0.0)),
            np.arctan2(y, x),
        )
        c, s = np.cos(t), np.sin(t)
        p = np.where(np.abs(c) >= 0.1, np.arctan2(z, x / c), np.arctan2(z, y / s))
        flip = p < 0.0
        p = np.where(flip, -p, p)
        t = np.where(flip, t - pi, t)
        t = np.where(t < 0.0, t + 2 * pi, t)
        t = np.where(vert, np.where(z < 0.0, pi, 0.0), t)
        p = np.where(vert, 0.5 * pi, p)
    return np.stack([t, p], -1)


def trend_plunge_to_vector(trend, plunge):
    """Unit vector (north, east, down) from trend/plunge in radians."""
    trend = np.asarray(trend)
    plunge = np.asarray(plunge)
    return np.stack(
        [np.cos(trend) * np.cos(plunge), np.sin(trend) * np.cos(plunge), np.sin(plunge)], -1
    )


def pt_axes(a, n):
    """Unit P and T axis vectors from A and N.

    Deliberate divergence from ``focreps.f``, which scales ``A -/+ N`` by ``sqrt(2)`` instead of
    ``1/sqrt(2)`` (``dsrin.f`` has it right).  The resulting vectors have length 2, so a
    component of exactly 1 (a true plunge of 30 degrees) made ``v2trpl`` report a vertical axis.
    """
    a = np.asarray(a)
    n = np.asarray(n)
    sr2 = a.dtype.type(1.0) / np.sqrt(a.dtype.type(2.0))  # stay in single precision for REAL*4 inputs
    return sr2 * (a - n), sr2 * (a + n)
