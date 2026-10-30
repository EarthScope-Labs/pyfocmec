"""Theoretical amplitude ratios for a double-couple source (port of ``lratio.f``)."""

from __future__ import annotations

import numpy as np

from .geometry import dsr_to_an

FLAG_NONE, FLAG_NUM, FLAG_DEN, FLAG_ND = 0, 1, 2, 3
FLAG_TEXT = {FLAG_NONE: "   ", FLAG_NUM: "NUM", FLAG_DEN: "DEN", FLAG_ND: "N&D"}


def log_ratio(jr, dip, strike, rake, xyz, xyzden, vpvs3, cutp, cuts):
    """Theoretical log10 amplitude ratio for many mechanisms at one station.

    ``jr`` is 1 for SV/P, 2 for SH/P and 3 for SV/SH.  ``dip``, ``strike`` and
    ``rake`` are arrays in radians; ``xyz`` (9) and ``xyzden`` (6) are the
    numerator/denominator direction vectors for the station.

    Returns ``(lograt, top, flag)`` where ``top`` carries the sign of the S
    polarity and ``flag`` is one of the ``FLAG_*`` codes.
    """
    a, n = dsr_to_an(dip, strike, rake)
    xyz = np.asarray(xyz)
    xyzden = np.asarray(xyzden)
    ran = a @ xyz[0:3]
    rnn = n @ xyz[0:3]
    tan = a @ xyz[3:6]
    tnn = n @ xyz[3:6]
    rad = a @ xyzden[0:3]
    rnd = n @ xyzden[0:3]
    tad = a @ xyzden[3:6]  # noqa: F841  (kept for parity with the Fortran)
    tnd = n @ xyzden[3:6]  # noqa: F841
    pa = a @ xyz[6:9]
    pn = n @ xyz[6:9]
    if jr < 3:
        bot = 2 * rad * rnd
    else:
        bot = rad * pn + rnd * pa
    if jr == 2:
        top = ran * pn + rnn * pa
    else:
        top = ran * tnn + rnn * tan

    cutbot = cutp if jr != 3 else cuts
    bot_small = np.abs(bot) <= cutbot
    top_small = np.abs(top) <= cuts
    nd = bot_small & top_small
    den = bot_small & ~top_small
    num = ~bot_small & top_small
    flag = np.zeros(np.shape(bot), dtype=np.int8)
    flag[num] = FLAG_NUM
    flag[den] = FLAG_DEN
    flag[nd] = FLAG_ND

    top_used = np.where(num, cuts * np.where(top < 0, -1.0, 1.0), top)
    bot_used = np.where(den, cutbot, bot)
    with np.errstate(all="ignore"):
        ratio = np.abs(top_used / bot_used)
        if jr != 3:
            ratio = vpvs3 * ratio
        lograt = np.log10(ratio)
    lograt = np.where(nd, 0.0, lograt)
    return lograt, top_used, flag


def radiation_pattern(dip, strike, rake, azimuth, ptoa, stoa):
    """P, SV and SH radiation factors of a double couple (port of ``radpat.f``).

    All angles are in degrees; ``azimuth`` is the station azimuth from the
    source and ``ptoa``/``stoa`` the P and S take-off angles measured from down.
    Sign conventions: P up, SV toward the station (down on the vertical), SH to
    the left looking at the station, are positive.  Arrays broadcast.
    """
    d, st, sl, az, ap, as_ = (np.radians(np.asarray(v, dtype=float)) for v in
                              (dip, strike, rake, azimuth, ptoa, stoa))
    a, n = dsr_to_an(d, st, sl)
    # unit vectors: R = radial, TH = SV direction, PH = SH direction
    r_p = np.stack([np.sin(ap) * np.cos(az), np.sin(ap) * np.sin(az), np.cos(ap)], -1)
    r_s = np.stack([np.sin(as_) * np.cos(az), np.sin(as_) * np.sin(az), np.cos(as_)], -1)
    th_s = np.stack([-np.cos(as_) * np.cos(az), -np.cos(as_) * np.sin(az), np.sin(as_)], -1)
    ph = np.stack([np.sin(az), -np.cos(az), np.zeros_like(az)], -1)
    dot = lambda u, v: np.sum(u * v, axis=-1)  # noqa: E731
    pr = 2.0 * dot(r_p, a) * dot(r_p, n)
    svr = dot(r_s, n) * dot(th_s, a) + dot(r_s, a) * dot(th_s, n)
    shr = dot(r_s, n) * dot(ph, a) + dot(r_s, a) * dot(ph, n)
    return pr, svr, shr


def free_surface(ips, vsvp, emergence, dtype=np.float64):
    """Free-surface amplification of an incident P (``ips=1``) or SV (``ips=2``) wave.

    Port of ``grndmtn.f``.  ``vsvp`` is Vs/Vp at the surface and ``emergence``
    the emergence angle in degrees from the vertical.  Returns
    ``(AR, AV, PhR, PhV)``: radial and vertical amplitude factors and their
    phases in degrees (non-zero only for post-critical SV).
    """
    dt = np.dtype(dtype).type
    rd = dt(45.0) / np.arctan(dt(1.0))
    vrat, aemrg = dt(vsvp), dt(emergence)
    mb = lambda x: 1 - dt(2.0) * x * x  # noqa: E731
    phr = phv = dt(0.0)
    if aemrg == 0.0:
        ar = dt(2.0) * (ips - 1)
        phr = dt((ips - 1) * 180.0)
        av = dt(2.0) * (2 - ips)
    elif aemrg == 90.0:
        ar = av = dt(0.0)
    elif aemrg == 45.0 and ips == 2:
        phr = dt(-90.0)
        av = np.sqrt(dt(2.0))
        ar = dt(0.0)
    elif ips == 1:
        tp = aemrg / rd
        ts = np.arcsin(np.sin(tp) * vrat)
        sts = np.sin(ts)
        mbsts = mb(sts)
        ctp, stp, cts = np.cos(tp), np.sin(tp), np.cos(ts)
        anum1 = dt(4.0) * sts * sts * ctp * cts * vrat
        den = anum1 + mbsts**2
        a = (anum1 - mbsts**2) / den
        b = dt(4.0) * sts * ctp * mbsts / den
        ar = stp * (1.0 + a) + cts * b
        if ar < 0.00001:
            ar = dt(0.0)
        av = ctp * (1.0 - a) + sts * b
        if av < 0.00002:
            av = dt(0.0)
    else:
        ts = aemrg / rd
        sts = np.sin(ts)
        mbsts = mb(sts)
        d2 = mbsts**2
        cts = np.cos(ts)
        anum1 = dt(4.0) * sts * cts * vrat
        anum = -anum1 * mbsts
        stp = sts / vrat
        if stp <= 1.0:
            tp = np.arcsin(stp)
            ctp = np.cos(tp)
            d1 = sts * ctp * anum1
            den = d1 + d2
            ar = (dt(2.0) * cts * d2 - stp * anum) / den
            phr = dt(180.0)
            av = (dt(2.0) * d1 * sts - ctp * anum) / den
            if abs(av) < 0.0001:
                av = dt(0.0)
        else:  # post-critical: complex reflection coefficients
            ctp = np.sqrt(stp * stp - 1.0)
            d1 = sts * ctp * anum1
            den = np.sqrt(d1 * d1 + d2 * d2)
            a = anum / den
            if abs(a) < 0.0001:
                a = dt(0.0)
            ar = (-dt(2.0) * d2 * cts + anum * stp) / den
            if abs(ar) < 0.0001:
                ar = dt(0.0)
            if ar < 0.0:
                ar = -ar
                phr = rd * np.arctan2(d1, -d2)
            else:
                phr = rd * np.arctan2(-d1, d2)
            av = (dt(2.0) * d1 * sts - anum * ctp) / den
            if av < 0.0:
                av = -av
                phv = rd * np.arctan2(d1, -d2) + 90.0
            else:
                phv = rd * np.arctan2(-d1, d2) + 90.0
    return float(ar), float(av), float(phr), float(phv)


def critical_angle(vsvp):
    """Critical emergence angle (degrees) for SV at a free surface."""
    return float(np.degrees(np.arcsin(vsvp)))


def free_surface_corrections(vsvp, p_emergence, s_emergence):
    """Free-surface amplitude corrections used for ratio data (``freesurf.f``).

    ``p_emergence`` and ``s_emergence`` are in degrees from the vertical.  The
    result is a dict of the amplitude factors and the correction multipliers
    that turn observed vertical/radial/transverse ratios into the SV/P, SH/P and
    SV/SH ratios of the focal mechanism.
    """
    if not 0.0 < vsvp < 1.0:
        raise ValueError("Vs/Vp must be between 0 and 1")
    ash = 2.0
    apr, apv = free_surface(1, vsvp, p_emergence)[:2]
    asr, asv = free_surface(2, vsvp, s_emergence)[:2]
    out = {
        "critical_angle": critical_angle(vsvp),
        "P_Z": apv, "P_R": apr, "S_Z": asv, "S_R": asr,
        "SH_T/P_Z": apv / ash,
        "supercritical": s_emergence > critical_angle(vsvp),
    }
    if asr != 0.0:
        out["SV_R/P_Z"] = apv / asr
        out["SV_R/SH_T"] = ash / asr
    if asv != 0.0:
        out["SV_Z/P_Z"] = apv / asv
        out["SV_Z/SH_T"] = ash / asv
        out["SV_R/SV_Z"] = asr / asv
    return out
