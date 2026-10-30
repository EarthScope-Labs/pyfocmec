"""Synthetic FOCMEC input for a known mechanism (port of ``synth_test.f``)."""

from __future__ import annotations

import numpy as np

from .inp import format_polarity, format_ratio
from .radiation import radiation_pattern


def synthetic_inp(dip, strike, rake, ptoa, stoa, vpvs=1.7321, n_azimuths=15, comment=None):
    """Lines of a FOCMEC ``.inp`` file with polarities and ratios for a mechanism.

    ``n_azimuths`` stations are spread evenly in azimuth (24 degrees apart for
    the default of 15) with the given P and S take-off angles.  Unlike the
    original program, the ratio lines carry the Vp/Vs field of the current
    input format.

    Returns ``(comment, lines)``.
    """
    if comment is None:
        comment = f"dip{dip:9.2f}; strike{strike:10.2f}; slip{rake:10.2f}; vP/vS{vpvs:10.4f}"  # synth_test.f format 121
    vpvs3 = vpvs**3
    lines = []
    step = 360.0 / n_azimuths
    for j in range(1, n_azimuths + 1):
        azd = (j - 1) * step
        rp, rsv, rsh = (float(x) for x in radiation_pattern(dip, strike, rake, azd, ptoa, stoa))
        sta = f"XX{j:02d}"
        if abs(rp) > 1e-3:
            lines.append(format_polarity(sta, azd, ptoa, "C" if rp >= 0 else "D"))
        if abs(rsv) > 1e-3:
            lines.append(format_polarity(sta, azd, stoa, "F" if rsv >= 0 else "B"))
        if abs(rsh) > 1e-3:
            lines.append(format_polarity(sta, azd, stoa, "L" if rsh >= 0 else "R"))
        if abs(rsv) > 1e-3 and abs(rp) > 1e-3:
            lines.append(format_ratio(sta, azd, stoa, "V", np.log10(abs(rsv / rp) * vpvs3),
                                      "F" if rsv >= 0 else "B", ptoa, vpvs, "sv/p"))
        if abs(rsh) > 1e-3 and abs(rp) > 1e-3:
            lines.append(format_ratio(sta, azd, stoa, "H", np.log10(abs(rsh / rp) * vpvs3),
                                      "L" if rsh >= 0 else "R", ptoa, vpvs, "sh/p"))
        if abs(rsh) > 1e-3 and abs(rsv) > 1e-3:
            lines.append(format_ratio(sta, azd, stoa, "S", np.log10(abs(rsv / rsh)),
                                      "F" if rsv >= 0 else "B", stoa, vpvs, "sv/sh"))
    return comment, lines
