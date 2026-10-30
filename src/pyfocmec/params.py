"""Search parameters (the answers FOCMEC asks for interactively)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def nint(x: float) -> int:
    """Fortran NINT: round half away from zero."""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


@dataclass
class FocmecParams:
    """User-facing parameters for a FOCMEC run.

    Polarity error limits are *weighted* when ``relweight`` is true (the
    default, "RW"): each misfit counts with the theoretical radiation factor at
    that station.  With ``relweight=False`` ("UW") every misfit counts 1.

    ``total_errors`` is the "total polarity error" option, which is only
    meaningful with both P and S polarities; otherwise the per-phase limits
    ``p_errors``, ``sv_errors`` and ``sh_errors`` are used.

    ``trend``, ``plunge`` and ``angle`` are ``(minimum, increment, maximum)``
    in degrees for the B-axis trend, B-axis plunge and the "Angle" of the A
    axis in the plane perpendicular to B.  A ``None`` maximum takes the
    FOCMEC default (360-inc, 90 and 180-inc).
    """

    relweight: bool = True
    total_errors: float | None = None
    p_errors: float = 0.0
    sv_errors: float = 0.0
    sh_errors: float = 0.0
    ratio_error: float = 0.6
    n_ratio_errors: int = 0
    cutp: float = 0.1
    cuts: float = 0.1
    maxsol: int = 100
    trend: tuple = (0.0, 5.0, None)
    plunge: tuple = (0.0, 5.0, 90.0)
    angle: tuple = (0.0, 5.0, None)
    comment: str = ""
    out_name: str = "focmec.out"
    emergent: bool | None = None


@dataclass
class Resolved:
    """Parameters after the clamping and bookkeeping done in ``focinp.f``."""

    relweight: bool
    thresh: float
    err: float
    errp: float
    errsv: float
    errsh: float
    total_mode: bool
    nppol: int
    nsvpol: int
    nshpol: int
    errrat: float
    nerrr: int
    cutp: float
    cuts: float
    vpvs3: float
    maxsol: int
    bt: tuple
    bp: tuple
    aa: tuple
    params: FocmecParams = field(repr=False, default=None)


def resolve(data, p: FocmecParams) -> Resolved:
    nppol, nsvpol, nshpol = (data.count_pol(k) for k in (0, 1, 2))
    npol = data.npol
    thresh = 0.0 if p.relweight else 1.0
    err = errp = errsv = errsh = 0.0
    total_mode = False
    if npol > 0:
        mixed = nppol > 0 and (nshpol > 0 or nsvpol > 0)
        if mixed and p.total_errors is not None:
            total_mode = True
            err = float(p.total_errors) if p.relweight else float(int(p.total_errors))
            errp = errsv = errsh = err
        else:
            tot = p.total_errors

            def lim(given, n):
                if n <= 0:
                    return 0.0
                v = given if (given or tot is None) else tot
                v = float(v) if p.relweight else float(nint(v))
                return min(v, float(n))

            errp = lim(p.p_errors, nppol)
            errsv = lim(p.sv_errors, nsvpol)
            errsh = lim(p.sh_errors, nshpol)
            err = errp + errsv + errsh
    nerrr = min(int(p.n_ratio_errors), data.nrat) if data.nrat else int(p.n_ratio_errors)

    btmin, btdel, btmax = p.trend
    btdel = abs(btdel)
    btmax = 360 - btdel if btmax is None else btmax
    btmax = max(btmin, min(btmax, 360 - btdel))
    bpmin, bpdel, bpmax = p.plunge
    bpdel = abs(bpdel)
    bpmax = 90.0 if bpmax is None else bpmax
    bpmax = max(bpmin, min(bpmax, 90.0))
    aamin, aadel, aamax = p.angle
    aadel = abs(aadel)
    aamax = 180 - aadel if aamax is None else aamax
    aamax = max(aamin, min(aamax, 180.0 - aadel))
    for step in (btdel, bpdel, aadel):
        if step <= 0:
            raise ValueError("search increments must be positive")
    return Resolved(
        relweight=p.relweight,
        thresh=thresh,
        err=err,
        errp=errp,
        errsv=errsv,
        errsh=errsh,
        total_mode=total_mode,
        nppol=nppol,
        nsvpol=nsvpol,
        nshpol=nshpol,
        errrat=p.ratio_error,
        nerrr=nerrr,
        cutp=p.cutp,
        cuts=p.cuts,
        vpvs3=data.vpvs**3,
        maxsol=int(p.maxsol),
        bt=(btmin, btdel, btmax),
        bp=(bpmin, bpdel, bpmax),
        aa=(aamin, aadel, aamax),
        params=p,
    )
