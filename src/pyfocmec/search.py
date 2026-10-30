"""Systematic focal-sphere search (port of ``srchfm.f``, ``oksol.f``, ``okpol.f``, ``okrat.f``).

The search is vectorised with numpy but visits trial mechanisms in exactly the
order of the Fortran loops (B plunge, then B trend, then Angle), so the first
``maxsol`` acceptable solutions are the same ones the original reports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .geometry import (
    an_to_dsr,
    fault_vectors,
    pt_axes,
    snap,
    vector_to_trend_plunge,
)
from .inp import FocmecData
from .params import FocmecParams, Resolved, nint, resolve
from .radiation import FLAG_ND, log_ratio

_JR = {"V": 1, "H": 2, "S": 3}


@dataclass
class Solution:
    """One acceptable mechanism (angles in degrees)."""

    index: int
    dip: float
    strike: float
    rake: float
    aux_dip: float
    aux_strike: float
    aux_rake: float
    a_trend: float
    a_plunge: float
    n_trend: float
    n_plunge: float
    p_trend: float
    p_plunge: float
    t_trend: float
    t_plunge: float
    b_trend: float
    b_plunge: float
    angle: float
    bad_p: float
    bad_sv: float
    bad_sh: float
    bad_stations: dict  # {"P": [(sta, weight), ...], "SV": [...], "SH": [...]}
    ratio_ok: int | None = None
    ratio_total: int | None = None
    ratio_rms: float | None = None
    ratio_maxdiff: float | None = None
    ratios: list | None = None  # per-ratio dicts


class SolutionSet:
    """Container of accepted solutions held as arrays."""

    def __init__(self, data: FocmecData, res: Resolved, dtype=np.float64):
        self.data = data
        self.res = res
        self.dtype = np.dtype(dtype)
        self.trend = np.zeros(0)
        self.plunge = np.zeros(0)
        self.angle = np.zeros(0)
        self.a = np.zeros((0, 3))
        self.n = np.zeros((0, 3))
        self.dsr = np.zeros((0, 3))
        self.badtot = np.zeros((0, 3))  # P, SV, SH weighted totals
        self.pol_w = np.zeros((0, data.npol))
        self.pol_bad = np.zeros((0, data.npol), dtype=bool)
        nr = data.nrat
        self.cal = np.zeros((0, nr))
        self.diff = np.zeros((0, nr))
        self.spol = np.zeros((0, nr))
        self.flag = np.zeros((0, nr), dtype=np.int8)
        self.wtrat = np.zeros((0, nr), dtype=int)
        self.nbadr = np.zeros(0, dtype=int)
        self.esum = np.zeros(0)
        self.emax = np.zeros(0)
        self.nallowed = np.zeros(0, dtype=int)

    def __len__(self):
        return len(self.trend)

    def _append(self, **kw):
        for k, v in kw.items():
            setattr(self, k, np.concatenate([getattr(self, k), v]))

    def __getitem__(self, i) -> Solution:
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        return _make_solution(self, i)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def to_records(self) -> list[dict]:
        """Rows as plain dicts (handy for ``pandas.DataFrame``)."""
        rows = []
        for s in self:
            rows.append(
                dict(
                    dip=s.dip, strike=s.strike, rake=s.rake,
                    aux_dip=s.aux_dip, aux_strike=s.aux_strike, aux_rake=s.aux_rake,
                    p_trend=s.p_trend, p_plunge=s.p_plunge, t_trend=s.t_trend, t_plunge=s.t_plunge,
                    b_trend=s.b_trend, b_plunge=s.b_plunge, angle=s.angle,
                    bad_p=s.bad_p, bad_sv=s.bad_sv, bad_sh=s.bad_sh,
                    ratio_ok=s.ratio_ok, ratio_total=s.ratio_total,
                    ratio_rms=s.ratio_rms, ratio_maxdiff=s.ratio_maxdiff,
                )
            )
        return rows


@dataclass
class FocmecResult:
    data: FocmecData
    resolved: Resolved
    solutions: SolutionSet
    truncated: bool
    timestamp: str

    @property
    def nsol(self) -> int:
        return len(self.solutions)

    @property
    def params(self) -> FocmecParams:
        return self.resolved.params

    def write_lst(self, path_or_file="focmec.lst", **kw):
        from .report import write_lst
        return write_lst(self, path_or_file, **kw)

    def write_out(self, path_or_file="focmec.out", **kw):
        from .report import write_out
        return write_out(self, path_or_file, **kw)


def _tp_deg(v):
    return np.degrees(vector_to_trend_plunge(v))


def _make_solution(ss: SolutionSet, i: int) -> Solution:
    d = ss.data
    dt = ss.dtype.type
    a, n = ss.a[i].astype(dt), ss.n[i].astype(dt)
    dsr = np.degrees(ss.dsr[i].astype(dt))
    aux = np.degrees(an_to_dsr(n, a))
    atp = _tp_deg(a.copy())
    ntp = _tp_deg(n.copy())
    pv, tv = pt_axes(a, n)
    ptp = _tp_deg(pv)
    ttp = _tp_deg(tv)
    bad = {"P": [], "SV": [], "SH": []}
    names = ("P", "SV", "SH")
    for k in range(d.npol):
        if ss.pol_bad[i, k]:
            bad[names[d.pol_kind[k]]].append((d.pol_sta[k], float(ss.pol_w[i, k])))
    sol = Solution(
        index=i,
        dip=dsr[0], strike=dsr[1], rake=dsr[2],
        aux_dip=aux[0], aux_strike=aux[1], aux_rake=aux[2],
        a_trend=atp[0], a_plunge=atp[1], n_trend=ntp[0], n_plunge=ntp[1],
        p_trend=ptp[0], p_plunge=ptp[1], t_trend=ttp[0], t_plunge=ttp[1],
        b_trend=float(ss.trend[i]), b_plunge=float(ss.plunge[i]), angle=float(ss.angle[i]),
        bad_p=float(ss.badtot[i, 0]), bad_sv=float(ss.badtot[i, 1]), bad_sh=float(ss.badtot[i, 2]),
        bad_stations=bad,
    )
    if d.nrat > 0:
        nbadr = int(ss.nbadr[i])
        sol.ratio_ok = d.nrat - nbadr
        sol.ratio_total = int(ss.nallowed[i])
        sol.ratio_rms = float(np.sqrt(ss.esum[i] / (d.nrat - nbadr)))
        sol.ratio_maxdiff = float(ss.emax[i])
        sol.ratios = [
            dict(
                station=d.rat_sta[k], sense=d.rat_sense[k], numpol=d.rat_numpol[k],
                observed=float(d.rat_log[k]), calculated=float(ss.cal[i, k]),
                difference=float(ss.diff[i, k]), flag=int(ss.flag[i, k]),
                weight=int(ss.wtrat[i, k]), spol=float(ss.spol[i, k]),
            )
            for k in range(d.nrat)
        ]
    return sol


_F4 = np.float32
_RD32 = _F4(45.0) / np.arctan(_F4(1.0))


def _grid(res: Resolved):
    """Yield (trend, plunge, angle) degree arrays row by row in srchfm.f order.

    The grid is always built in single precision with the same operation order as
    ``srchfm.f``: the number of trend steps is a NINT of a REAL*4 ratio, so exact
    ties (k + 0.5) must round the way the original does.
    """
    btmin, btdel, btmax = map(_F4, res.bt)
    bpmin, bpdel, bpmax = map(_F4, res.bp)
    aamin, aadel, aamax = map(_F4, res.aa)
    nplung = 1 + nint(float((bpmax - bpmin) / bpdel))
    if nplung == 1:
        bpmax = bpmin
    naang = 1 + nint(float((aamax - aamin) / aadel))
    if naang == 1:
        aamax = aamin
    ang = np.array([aamin + _F4(ja) * aadel for ja in range(naang)], dtype=_F4)
    ang[-1] = aamax  # the last step always lands exactly on the maximum
    for jp in range(1, nplung + 1):
        plunge = (bpmin + _F4(jp - 1) * bpdel) if jp < nplung else bpmax
        prad = _F4(plunge / _RD32)
        if jp == nplung and bpmax >= 90.0:
            ntrend, btdeln = 1, _F4(0.0)
        else:
            btdeln = _F4(btdel / np.cos(prad))
            ntrend = nint(float(_F4(btmax + btdel - btmin) / btdeln))
            if ntrend > 0:
                btdeln = _F4(_F4(btmax + btdel - btmin) / _F4(ntrend))
        if ntrend <= 0:
            continue
        trends = np.array(
            [min(_F4(btmin + _F4(jt) * btdeln), btmax) for jt in range(ntrend)], dtype=_F4
        )
        if prad == 0.0:
            trends = trends[(trends / _RD32) < _F4(_F4(180.0) / _RD32)]
        if trends.size == 0:
            continue
        t = np.repeat(trends, naang)
        a = np.tile(ang, trends.size)
        yield t, np.full(t.shape, plunge, dtype=_F4), a


def _rad(deg, dt):
    """Degrees to radians; in single precision divide by a REAL*4 RD like the Fortran."""
    if dt is np.float32:
        return (np.asarray(deg, dtype=_F4) / _RD32).astype(_F4)
    return np.radians(deg).astype(dt)


def _chunks(res: Resolved, target: int):
    buf_t, buf_p, buf_a, n = [], [], [], 0
    for t, p, a in _grid(res):
        buf_t.append(t), buf_p.append(p), buf_a.append(a)
        n += t.size
        if n >= target:
            yield np.concatenate(buf_t), np.concatenate(buf_p), np.concatenate(buf_a)
            buf_t, buf_p, buf_a, n = [], [], [], 0
    if n:
        yield np.concatenate(buf_t), np.concatenate(buf_p), np.concatenate(buf_a)


def _dot3(v, x):
    """Dot product of rows of ``v`` (m,3) with ``x`` (3,), summed term by term like the Fortran loops."""
    return v[:, 0] * x[0] + v[:, 1] * x[1] + v[:, 2] * x[2]


def _polarity(data: FocmecData, res: Resolved, a, n, dt, keep_detail=False):
    """Port of okpol: returns (ok, totals(m,3), weights(m,npol), bad(m,npol))."""
    m = a.shape[0]
    bad_all = np.zeros(m, dtype=dt)
    tot = np.zeros((m, 3), dtype=dt)
    npol = data.npol
    if keep_detail:
        w_store = np.zeros((m, npol), dtype=dt)
        b_store = np.zeros((m, npol), dtype=bool)
    thresh = dt(res.thresh)
    for k in range(npol):
        xyz = (data.pol_xyz32 if dt is np.float32 else data.pol_xyz)[k].astype(dt)
        kind = data.pol_kind[k]
        ra = _dot3(a, xyz[0:3])
        rn = _dot3(n, xyz[0:3])
        if kind == 0:
            test = dt(data.pol_sign[k]) * 2 * ra * rn
        elif kind == 1:
            ta = _dot3(a, xyz[3:6])
            tn = _dot3(n, xyz[3:6])
            test = dt(data.pol_sign[k]) * (ra * tn + rn * ta)
        else:
            pa = _dot3(a, xyz[6:9])
            pn = _dot3(n, xyz[6:9])
            test = dt(data.pol_sign[k]) * (ra * pn + rn * pa)
        bad = test <= 0.0
        w = np.where(bad, np.maximum(-test, thresh), 0.0).astype(dt)
        bad_all = bad_all + w
        tot[:, kind] = tot[:, kind] + w
        if keep_detail:
            w_store[:, k] = w
            b_store[:, k] = bad
    ok = (
        (bad_all <= res.err)
        & (tot[:, 0] <= res.errp)
        & (tot[:, 1] <= res.errsv)
        & (tot[:, 2] <= res.errsh)
    )
    if keep_detail:
        return ok, tot, w_store, b_store
    return ok, tot, None, None


def _ratios(data: FocmecData, res: Resolved, dsr, dt):
    """Port of okrat for many mechanisms; returns dict of arrays."""
    m = dsr.shape[0]
    nr = data.nrat
    cal = np.zeros((m, nr), dtype=dt)
    diff = np.zeros((m, nr), dtype=dt)
    spol = np.zeros((m, nr), dtype=dt)
    flag = np.zeros((m, nr), dtype=np.int8)
    wt = np.zeros((m, nr), dtype=int)
    nbadr = np.zeros(m, dtype=int)
    esum = np.zeros(m, dtype=dt)
    emax = np.zeros(m, dtype=dt)
    nd = np.zeros(m, dtype=int)
    for k in range(nr):
        lr, top, fl = log_ratio(
            _JR[data.rat_sense[k]], dsr[:, 0], dsr[:, 1], dsr[:, 2],
            (data.rat_xyz32 if dt is np.float32 else data.rat_xyz)[k].astype(dt),
            (data.rat_xyzden32 if dt is np.float32 else data.rat_xyzden)[k].astype(dt),
            dt(res.vpvs3), dt(res.cutp), dt(res.cuts),
        )
        isnd = fl == FLAG_ND
        d = (dt(data.rat_log[k]) - lr).astype(dt)
        d = np.where(isnd, 0.0, d).astype(dt)
        toobig = ~isnd & (np.abs(d) > res.errrat)
        good = ~isnd & ~toobig
        nbadr += (isnd | toobig).astype(int)
        nd += isnd.astype(int)
        esum = esum + np.where(good, d * d, 0.0).astype(dt)
        emax = np.where(good & (np.abs(d) > emax), np.abs(d), emax).astype(dt)
        cal[:, k], diff[:, k], spol[:, k], flag[:, k], wt[:, k] = lr, d, top, fl, good
    ok = (nbadr <= res.nerrr) & ((nr - nd) > nbadr)
    return dict(ok=ok, cal=cal, diff=diff, spol=spol, flag=flag, wtrat=wt, nbadr=nbadr,
                esum=esum, emax=emax, nallowed=nr - nd)


def search(data: FocmecData, params: FocmecParams | None = None, dtype=np.float64) -> FocmecResult:
    """Run the FOCMEC grid search and return all acceptable solutions.

    ``dtype=np.float32`` mimics the single-precision arithmetic of the original
    program (useful when comparing with reference output near thresholds).
    """
    params = params or FocmecParams()
    res = resolve(data, params)
    dt = np.dtype(dtype).type
    ss = SolutionSet(data, res, dtype)
    maxsol = res.maxsol
    truncated = False
    target = int(min(200000, max(2000, 1_000_000 // max(data.nrat, data.npol, 1))))
    nsol = 0
    for t, p, ang in _chunks(res, target):
        a, n = fault_vectors(_rad(p, dt), _rad(t, dt), _rad(ang, dt))
        if data.npol > 0:
            ok, tot, _, _ = _polarity(data, res, a, n, dt)
        else:
            ok = np.ones(a.shape[0], dtype=bool)
            tot = np.zeros((a.shape[0], 3), dtype=dt)
        idx = np.nonzero(ok)[0]
        if idx.size == 0:
            continue
        a, n, t, p, ang, tot = a[idx], n[idx], t[idx], p[idx], ang[idx], tot[idx]
        a_raw, n_raw = a, n  # okpol reports bad stations from the vectors before they are snapped
        a = snap(a).astype(dt)
        n = snap(n).astype(dt)
        dsr = an_to_dsr(a, n).astype(dt)
        if data.nrat > 0:
            r = _ratios(data, res, dsr, dt)
            keep = np.nonzero(r["ok"])[0]
            if keep.size == 0:
                continue
        else:
            r = None
            keep = np.arange(a.shape[0])
        if maxsol > 0 and nsol + keep.size >= maxsol:
            keep = keep[: maxsol - nsol]
            truncated = True
        a, n, t, p, ang, tot, dsr = (x[keep] for x in (a, n, t, p, ang, tot, dsr))
        if data.npol > 0:
            _, _, w, b = _polarity(data, res, a_raw[keep], n_raw[keep], dt, keep_detail=True)
        else:
            w = np.zeros((a.shape[0], 0))
            b = np.zeros((a.shape[0], 0), dtype=bool)
        kw = dict(trend=t, plunge=p, angle=ang, a=a.astype(float), n=n.astype(float),
                  dsr=dsr.astype(float), badtot=tot.astype(float))
        ss._append(**kw)
        ss.pol_w = np.concatenate([ss.pol_w, w.astype(float)])
        ss.pol_bad = np.concatenate([ss.pol_bad, b])
        if r is not None:
            for key in ("cal", "diff", "spol", "flag", "wtrat", "nbadr", "esum", "emax", "nallowed"):
                setattr(ss, key, np.concatenate([getattr(ss, key), r[key][keep]]))
        nsol += keep.size
        if truncated:
            break
    stamp = time.strftime("%a %b ") + f"{time.localtime().tm_mday:2d}" + time.strftime(" %H:%M:%S %Y")
    return FocmecResult(data=data, resolved=res, solutions=ss, truncated=truncated, timestamp=stamp)
