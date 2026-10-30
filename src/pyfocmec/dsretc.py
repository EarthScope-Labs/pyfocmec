"""Conversions between focal-mechanism representations (port of ``dsretc.f`` + ``fmreps``).

Given dip/strike/rake, P and T axes, or a moment tensor, compute every other
representation (both nodal planes, A/N/B/P/T axes, the normalised double-couple
moment tensor), the FOCMEC search "Angle" for a fixed-solution run and an ASCII
beachball.  Angles are in degrees at the public interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .fortfmt import fnum
from .geometry import an_to_dsr, fault_vectors, vector_to_trend_plunge

# The original uses this truncated constant for 1/sqrt(2) in dsrin/pttpin.
SR2 = 0.707107


@dataclass
class Representations:
    """All representations of one double-couple mechanism (degrees)."""

    dsr: tuple  # dip, strike, rake of the plane implied by the input
    aux: tuple  # dip, strike, rake of the auxiliary plane
    a: tuple  # trend, plunge of A (slip vector)
    n: tuple  # trend, plunge of N (normal)
    b: tuple  # trend, plunge of B (null axis)
    p: tuple
    t: tuple
    moment: tuple  # MRR MTT MPP MRT MRP MTP of the unit double couple
    angle: float | None = None  # FOCMEC search "Angle" for this mechanism
    notes: list = field(default_factory=list)  # lines produced while reading a moment tensor

    @property
    def pttp(self):
        return (*self.p, *self.t)

    def lines(self) -> list[str]:
        """The block printed by ``fmreps.f`` (to dsretc.lst and the terminal)."""
        f9 = lambda v: "".join(fnum(x, 9, 2) for x in v)  # noqa: E731
        m = self.moment
        return [
            "     Dip,Strike,Rake " + f9(self.dsr),
            "     Dip,Strike,Rake " + f9(self.aux) + "   Auxiliary Plane",
            "     Lower Hem. Trend, Plunge of A,N " + f9(self.a + self.n),
            "     Lower Hem. Trend & Plunge of B " + f9(self.b),
            "     Lower Hem. Trend, Plunge of P,T " + f9(self.p + self.t),
            "     MRR =" + fnum(m[0], 5, 2) + "  MTT =" + fnum(m[1], 5, 2) + "  MPP =" + fnum(m[2], 5, 2)
            + "  MRT =" + fnum(m[3], 5, 2) + "  MRP =" + fnum(m[4], 5, 2) + "  MTP =" + fnum(m[5], 5, 2),
        ]


def _snap_dsrin(x):
    x = np.where(np.abs(x) <= 1e-4, 0.0, x)
    with np.errstate(all="ignore"):
        return np.where(np.abs(x) - 1.0 > 0.0, x / np.abs(x), x).astype(x.dtype)


def _v2snap(x):
    """The clean-up V2TRPL does to its argument in place (``v2trpl.f``), which callers rely on."""
    x = np.where(np.abs(x) <= 1e-4, 0.0, x)
    return np.where(np.abs(np.abs(x) - 1.0) < 1e-4, np.sign(x), x).astype(x.dtype)


def _trpl2v(trend, plunge, dt):
    trend, plunge = dt(trend), dt(plunge)
    v = np.array(
        [np.cos(trend) * np.cos(plunge), np.sin(trend) * np.cos(plunge), np.sin(plunge)], dtype=dt
    )
    v = np.where(np.abs(v) < 1e-4, 0.0, v).astype(dt)
    return np.where(np.abs(np.abs(v) - 1.0) < 1e-4, np.sign(v), v).astype(dt)


def an_to_moment(a, n):
    """Port of ``an2mom.f``: unit double-couple moment tensor (MRR MTT MPP MRT MRP MTP)."""
    m = np.array(
        [
            2.0 * a[2] * n[2],
            2.0 * a[0] * n[0],
            2.0 * a[1] * n[1],
            a[0] * n[2] + a[2] * n[0],
            -a[1] * n[2] - a[2] * n[1],
            -a[1] * n[0] - a[0] * n[1],
        ],
        dtype=a.dtype,
    )
    return np.where(np.abs(m) < 0.000001, 0.0, m).astype(a.dtype)


def _from_dsr_rad(dsr, dt):
    """Port of ``dsrin.f``; ``dsr`` in radians."""
    dip, strike, rake = (dt(v) for v in dsr)
    a = np.array(
        [
            np.cos(rake) * np.cos(strike) + np.sin(rake) * np.cos(dip) * np.sin(strike),
            np.cos(rake) * np.sin(strike) - np.sin(rake) * np.cos(dip) * np.cos(strike),
            -np.sin(rake) * np.sin(dip),
        ],
        dtype=dt,
    )
    n = np.array(
        [-np.sin(strike) * np.sin(dip), np.cos(strike) * np.sin(dip), -np.cos(dip)], dtype=dt
    )
    b = np.array(
        [
            np.cos(strike) * np.sin(rake) - np.cos(rake) * np.cos(dip) * np.sin(strike),
            np.cos(rake) * np.cos(strike) * np.cos(dip) + np.sin(rake) * np.sin(strike),
            np.cos(rake) * np.sin(dip),
        ],
        dtype=dt,
    )
    a, n, b = (_v2snap(_snap_dsrin(v)) for v in (a, n, b))  # dsrin.f: V2TRPL(A), (N), (B) snap in place
    tp = lambda v: np.degrees(vector_to_trend_plunge(v.copy()))  # noqa: E731
    sr2 = dt(SR2)
    t = sr2 * (a + n)
    p = sr2 * (a - n)
    aux = np.degrees(an_to_dsr(n, a))
    return Representations(
        dsr=tuple(float(x) for x in np.degrees(np.array([dip, strike, rake], dtype=dt))),
        aux=tuple(float(x) for x in aux),
        a=tuple(float(x) for x in tp(a)),
        n=tuple(float(x) for x in tp(n)),
        b=tuple(float(x) for x in tp(b)),
        p=tuple(float(x) for x in tp(p)),
        t=tuple(float(x) for x in tp(t)),
        moment=tuple(float(x) for x in an_to_moment(a, n)),
    )


def from_dsr(dip, strike, rake, dtype=np.float64) -> Representations:
    """Representations of the mechanism with this dip, strike and rake (degrees)."""
    dt = np.dtype(dtype).type
    rd = dt(45.0) / np.arctan(dt(1.0))
    rep = _from_dsr_rad(np.array([dip, strike, rake], dtype=dt) / rd, dt)
    rep.angle = get_angle(rep.dsr, dtype)
    return rep


def from_pt(p_trend, p_plunge, t_trend, t_plunge, dtype=np.float64) -> Representations:
    """Representations from P and T axis trend/plunge in degrees (``pttpin.f``)."""
    dt = np.dtype(dtype).type
    rd = dt(45.0) / np.arctan(dt(1.0))
    pt = np.array([p_trend, p_plunge, t_trend, t_plunge], dtype=dt) / rd
    pv = _trpl2v(pt[0], pt[1], dt)
    tv = _trpl2v(pt[2], pt[3], dt)
    sr2 = dt(SR2)
    a = sr2 * (pv + tv)
    n = sr2 * (tv - pv)
    dsr = an_to_dsr(a, n).astype(dt)
    rep = _from_dsr_rad(dsr, dt)
    rep.angle = get_angle(rep.dsr, dtype)
    return rep


# ---------------------------------------------------------------- moment tensors
def _sign(a, b):
    return abs(a) if b >= 0 else -abs(a)


def _tred2(a, dt):
    """EISPACK tred2 for a symmetric n x n matrix (list of lists, 0-based, ``dt`` scalars)."""
    n = len(a)
    z = [[dt(0.0)] * n for _ in range(n)]
    d = [dt(0.0)] * n
    e = [dt(0.0)] * n
    for i in range(n):
        for j in range(i, n):
            z[j][i] = a[j][i]
    for i in range(n):
        d[i] = a[n - 1][i]
    for i in range(n - 1, 0, -1):
        l = i - 1  # noqa: E741  (0-based; the Fortran L = i-1 in 1-based terms)
        h = dt(0.0)
        scale = dt(0.0)
        if l >= 1:
            for k in range(l + 1):
                scale = scale + abs(d[k])
        if l < 1 or scale == 0.0:
            e[i] = d[l]
            for j in range(l + 1):
                d[j] = z[l][j]
                z[i][j] = dt(0.0)
                z[j][i] = dt(0.0)
        else:
            for k in range(l + 1):
                d[k] = d[k] / scale
                h = h + d[k] * d[k]
            f = d[l]
            g = -_sign(np.sqrt(h), f)
            e[i] = scale * g
            h = h - f * g
            d[l] = f - g
            for j in range(l + 1):
                e[j] = dt(0.0)
            for j in range(l + 1):
                f = d[j]
                z[j][i] = f
                g = e[j] + z[j][j] * f
                for k in range(j + 1, l + 1):
                    g = g + z[k][j] * d[k]
                    e[k] = e[k] + z[k][j] * f
                e[j] = g
            f = dt(0.0)
            for j in range(l + 1):
                e[j] = e[j] / h
                f = f + e[j] * d[j]
            hh = -f / (h + h)
            for j in range(l + 1):
                e[j] = e[j] + hh * d[j]
            for j in range(l + 1):
                f = -d[j]
                g = -e[j]
                for k in range(j, l + 1):
                    z[k][j] = z[k][j] + f * e[k] + g * d[k]
                d[j] = z[l][j]
                z[i][j] = dt(0.0)
        d[i] = h
    for i in range(1, n):
        l = i - 1  # noqa: E741
        z[n - 1][l] = z[l][l]
        z[l][l] = dt(1.0)
        h = d[i]
        if h != 0.0:
            for k in range(l + 1):
                d[k] = z[k][i] / h
            for j in range(l + 1):
                g = dt(0.0)
                for k in range(l + 1):
                    g = g + z[k][i] * z[k][j]
                g = -g
                for k in range(l + 1):
                    z[k][j] = z[k][j] + g * d[k]
        for k in range(l + 1):
            z[k][i] = dt(0.0)
    for i in range(n):
        d[i] = z[n - 1][i]
        z[n - 1][i] = dt(0.0)
    z[n - 1][n - 1] = dt(1.0)
    e[0] = dt(0.0)
    return d, e, z


def _imtql2(d, e, z, dt):
    """EISPACK imtql2 (eigenvalues ascending in ``d``, eigenvectors in columns of ``z``)."""
    n = len(d)
    one, two = dt(1.0), dt(2.0)
    for i in range(1, n):
        e[i - 1] = e[i]
    e[n - 1] = dt(0.0)
    for l in range(n):  # noqa: E741
        j = 0
        while True:
            m = l
            for m in range(l, n - 1):
                tst1 = abs(d[m]) + abs(d[m + 1])
                tst2 = tst1 + abs(e[m])
                if tst2 == tst1:
                    break
            else:
                m = n - 1
            p = d[l]
            if m == l:
                break
            if j == 30:
                raise RuntimeError("imtql2: no convergence")
            j += 1
            g = (d[l + 1] - p) / (two * e[l])
            if abs(g) <= 1.0:
                r = np.sqrt(one + g * g)
            else:
                r = g * np.sqrt(one + (one / g) ** 2)
            g = d[m] - p + e[l] / (g + _sign(r, g))
            s = one
            c = one
            p = dt(0.0)
            restart = False
            for i in range(m - 1, l - 1, -1):
                f = s * e[i]
                b = c * e[i]
                if abs(f) >= abs(g):
                    r = abs(f) * np.sqrt(one + (g / f) ** 2)
                elif g != 0.0:
                    r = abs(g) * np.sqrt((f / g) ** 2 + one)
                else:
                    r = abs(f)
                e[i + 1] = r
                if r == 0.0:
                    d[i + 1] = d[i + 1] - p
                    e[m] = dt(0.0)
                    restart = True
                    break
                s = f / r
                c = g / r
                g = d[i + 1] - p
                r = (d[i] - g) * s + two * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b
                for k in range(n):
                    f = z[k][i + 1]
                    z[k][i + 1] = s * z[k][i] + c * f
                    z[k][i] = c * z[k][i] - s * f
            if restart:
                continue
            d[l] = d[l] - p
            e[l] = g
            e[m] = dt(0.0)
    for i in range(n - 1):
        k = i
        p = d[i]
        for j in range(i + 1, n):
            if d[j] >= p:
                continue
            k = j
            p = d[j]
        d[k] = d[i]
        d[i] = p
        for j in range(n):
            p = z[j][i]
            z[j][i] = z[j][k]
            z[j][k] = p


def g11_4(x) -> str:
    """Fortran ``1PG11.4``."""
    x = float(x)
    ax = abs(x)
    if math.isnan(x):
        return "NaN".rjust(11)
    if math.isinf(x):
        return ("-Infinity" if x < 0 else "Infinity").rjust(11)
    if ax == 0.0:
        return "  0.000" + " " * 4
    # the choice of F or E layout is made on the value rounded to 4 significant digits
    exp10 = int(f"{ax:.3e}".split("e")[1])
    if exp10 == -1:
        body = f"{x:7.4f}"
    elif exp10 == 0:
        body = f"{x:7.3f}"
    elif exp10 == 1:
        body = f"{x:7.2f}"
    elif exp10 == 2:
        body = f"{x:7.1f}"
    elif exp10 == 3:
        body = f"{x:6.0f}.".rjust(7)
    else:
        mant, exp = f"{x:.4E}".split("E")
        return f"{mant}E{int(exp):+03d}".rjust(11)
    return body + " " * 4


def from_moment_tensor(mrr, mtt, mpp, mrt, mrp, mtp, dtype=np.float32):
    """Closest double couple of a moment tensor (Dziewonski convention), ``mt_in.f``.

    Returns ``(Representations, notes)``; raises ``ValueError`` for purely
    isotropic sources or those with under 50 % double couple (as the original,
    which stops).
    """
    dt = np.dtype(dtype).type
    m = [dt(v) for v in (mrr, mtt, mpp, mrt, mrp, mtp)]
    a = [[dt(0.0)] * 3 for _ in range(3)]
    a[2][2] = m[0]
    a[0][0] = m[1]
    a[1][1] = m[2]
    a[0][2] = m[3]
    a[2][0] = a[0][2]
    a[1][2] = -m[4]
    a[2][1] = a[1][2]
    a[0][1] = -m[5]
    a[1][0] = a[0][1]
    notes = ["  Input is moment tensor (Dziewonski convention)", " MRR MTT MPP MRT MRP MTP"]
    # the Fortran write(2,*) lines carry one leading blank
    notes = [" " + notes[0], notes[1], "".join(g11_4(v) for v in m)]
    d, e, z = _tred2(a, dt)
    _imtql2(d, e, z, dt)
    w = d
    isotrop = dt(0.0)
    for j in range(3):
        isotrop = isotrop + w[j]
    devmom = dt(0.0)
    for j in range(3):
        w[j] = w[j] - isotrop / dt(3.0)
        with np.errstate(all="ignore"):
            devmom = devmom + w[j] * w[j]
    with np.errstate(all="ignore"):
        devmom = np.sqrt(dt(0.5) * devmom)
    if devmom < dt(0.001) * isotrop:
        devmom = dt(0.0)
    notes += [
        "  ",
        "  Trace of moment tensor = " + g11_4(isotrop) + "  Deviatoric tensor moment = " + g11_4(devmom),
    ]
    if devmom == 0.0:
        notes.append(" Exiting because purely isotropic source")
        raise ValueError("purely isotropic source", notes)
    notes.append("  Deviatoric moment tensor eigenvalues:" + "".join(g11_4(v) for v in w))
    eps = abs(w[1] / max(-w[0], w[2]))
    eps1 = eps * dt(200.0)
    eps2 = dt(100.0) - eps1
    notes.append("  EPSILON    % OF CLVD     % OF DC")
    notes.append(fnum(eps, 9, 4) + fnum(eps1, 12, 4) + fnum(eps2, 12, 4))
    notes.append("  ")
    if eps >= 0.25:
        notes.append("  Exiting because less than 50% double couple")
        raise ValueError("less than 50% double couple", notes)
    uvec = np.array([[z[r][c] for c in range(3)] for r in range(3)], dtype=dt)
    uvec[:, 0] = -uvec[:, 0]  # eig() flips the first eigenvector; matters for horizontal axes
    ptrp = lambda v: np.degrees(vector_to_trend_plunge(v.copy())).astype(dt)  # noqa: E731
    p = ptrp(uvec[:, 0])
    t = ptrp(uvec[:, 2])
    rep = from_pt(p[0], p[1], t[0], t[1], dtype)
    rep.notes = notes
    return rep, notes


# ---------------------------------------------------------------- Angle for FOCMEC
def get_angle(dsr, dtype=np.float64):
    """The FOCMEC search "Angle" (degrees) of the A axis for a given mechanism.

    Together with B trend and plunge (``Representations.b``) this defines a
    single grid point, so FOCMEC can be run on a fixed solution (port of
    ``get_angle`` in ``dsretc.f``).  Returns ``None`` if no consistent angle is found.
    """
    dt = np.dtype(dtype).type
    pi = dt(4.0) * np.arctan(dt(1.0))
    rd = dt(180.0) / pi
    dsr = np.array(dsr, dtype=dt) / rd
    dip, strike, rake = dsr
    a = np.array(
        [
            np.cos(rake) * np.cos(strike) + np.sin(rake) * np.cos(dip) * np.sin(strike),
            np.cos(rake) * np.sin(strike) - np.sin(rake) * np.cos(dip) * np.cos(strike),
            -np.sin(rake) * np.sin(dip),
        ],
        dtype=dt,
    )
    n = np.array([-np.sin(strike) * np.sin(dip), np.cos(strike) * np.sin(dip), -np.cos(dip)], dtype=dt)
    b = np.array(
        [
            np.cos(strike) * np.sin(rake) - np.cos(rake) * np.cos(dip) * np.sin(strike),
            np.cos(rake) * np.cos(strike) * np.cos(dip) + np.sin(rake) * np.sin(strike),
            np.cos(rake) * np.sin(dip),
        ],
        dtype=dt,
    )
    a, n, b = _v2snap(a), _v2snap(n), _v2snap(b)  # get_angle calls V2TRPL on A, N and B first
    bt, bp = vector_to_trend_plunge(b.copy())
    # first row of BMATRX = YROT @ ZROT
    ct, st = np.cos(bt), np.sin(bt)
    sp, cp = np.sin(bp), np.cos(bp)
    x = np.array([sp * ct, sp * st, -cp], dtype=dt)
    ain, nin = a.copy(), n.copy()

    def ok_ang(d1, d2):
        return bool(np.all(np.abs(d1 - d2) <= 0.001))

    for k in (1, 2):
        if k == 2:
            n, a = ain.copy(), nin.copy()
            dsr = an_to_dsr(a, n).astype(dt)
        cosangle = float(np.sum(n * x))
        angledeg = rd * np.arccos(dt(np.clip(cosangle, -1.0, 1.0)))
        for trial in (angledeg, dt(180.0) - angledeg):
            a_t, n_t = fault_vectors(np.array([bp], dtype=dt), np.array([bt], dtype=dt),
                                     np.array([trial / rd], dtype=dt))
            dsr2 = an_to_dsr(a_t[0], n_t[0]).astype(dt)
            if ok_ang(dsr, dsr2):
                return float(trial)
    return None


# ---------------------------------------------------------------- ASCII beachball
def ascii_beachball(moment, p, t, dtype=np.float32) -> list[str]:
    """Port of ``bball``: a 22-line text beachball (``#`` compressional, ``-`` dilatational)."""
    dt = np.dtype(dtype).type
    rd = dt(45.0) / np.arctan(dt(1.0))
    g = [dt(v) for v in moment]
    mxx, mxy, mxz, myy, myz, mzz = g[1], -g[5], g[3], g[2], -g[4], g[0]
    radius = dt(1.41)
    r0 = radius
    x0 = r0 + dt(0.250)
    y0 = r0 + dt(0.500)
    ix0 = int(dt(12.0) * x0)
    iy0 = int(dt(6.5) * y0)
    nx, ny = 2 * ix0, 2 * iy0
    ach = [[" "] * nx for _ in range(ny)]
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            dx = dt(i - ix0) / dt(12.0)
            dy = -dt(j - iy0) / dt(6.5)
            dd = dx * dx + dy * dy
            delta = np.sqrt(dd) if dd > 0 else dt(0.0)
            theta = rd * np.arctan2(dx, dy) if not (dx == 0 and dy == 0) else dt(0.0)
            if delta > r0:
                continue
            aoi = dt(90.0) if delta >= r0 else dt(90.0) * delta / r0
            x = np.cos(theta / rd) * np.sin(aoi / rd)
            y = np.sin(theta / rd) * np.sin(aoi / rd)
            z = np.cos(aoi / rd)
            pol = x * mxx * x + 2 * x * mxy * y + 2 * x * mxz * z + 2 * y * myz * z + y * myy * y + z * mzz * z
            ach[j - 1][i - 1] = "#" if pol > 0.0 else "-"

    def mark(trend, plunge, ch):
        trend, plunge = dt(trend), dt(plunge)
        ixp = int(math.floor(float(r0 * dt(12.0) * (dt(90.0) - plunge) * np.sin(trend / rd) / dt(90.0) + dt(ix0)) + 0.5))
        iyp = int(math.floor(float(-r0 * dt(6.5) * (dt(90.0) - plunge) * np.cos(trend / rd) / dt(90.0) + dt(iy0)) + 0.5))
        for i in range(ixp - 1, ixp + 2):
            for j in range(iyp - 1, iyp + 2):
                if 1 <= i <= nx and 1 <= j <= ny:
                    ach[j - 1][i - 1] = " "
        ach[iyp - 1][ixp - 1] = ch

    mark(p[0], p[1], "P")
    mark(t[0], t[1], "T")
    return ["     " + "".join(row) for row in ach[: 2 * iy0 - 2]]
