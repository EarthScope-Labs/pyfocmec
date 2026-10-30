"""Compare FOCMEC text output allowing last-digit differences from float arithmetic."""

import math
import re

_NUM = re.compile(r"[-+]?\d+\.\d+")
_PT = "Lower Hem. Trend, Plunge of P,T"


def _unit(trend, plunge):
    t, p = math.radians(trend), math.radians(plunge)
    return (math.cos(t) * math.cos(p), math.sin(t) * math.cos(p), math.sin(p))


def pt_axis_divergence(got: str, ref: str, tol: float = 0.0105) -> bool:
    """Known deliberate divergence from the Fortran (see CHANGELOG, "P/T axes").

    ``focreps.f`` builds P and T with ``sqrt(2)`` instead of ``1/sqrt(2)``, so when an axis has a
    true plunge of 30 degrees its z component is exactly 1 and ``v2trpl`` reports it as vertical.
    The reference then shows plunge 90.00 where pyfocmec shows 30.00.  Everything else on the
    line must agree and the two printed axes must be orthogonal.
    """
    if _PT not in got or _PT not in ref:
        return False
    g = [float(x) for x in _NUM.findall(got)]
    r = [float(x) for x in _NUM.findall(ref)]
    if len(g) != 4 or len(r) != 4:
        return False
    fixed = False
    for k in (0, 2):  # P then T: (trend, plunge) pairs
        if abs(g[k + 1] - r[k + 1]) <= tol and (abs(g[k] - r[k]) <= tol or abs(abs(g[k] - r[k]) - 360.0) <= tol):
            continue
        if abs(r[k + 1] - 90.0) <= tol and abs(g[k + 1] - 30.0) <= tol:
            fixed = True
            continue
        return False
    if not fixed:
        return False
    dot = sum(x * y for x, y in zip(_unit(g[0], g[1]), _unit(g[2], g[3])))
    return abs(dot) < 2e-3


def lines_match(got: str, ref: str, tol: float = 0.0105) -> bool:
    """Same text and column layout; numbers may differ by a last printed digit (``tol``).

    Whitespace and number widths must be identical, and a trend of 360.00 is not accepted for 0.00:
    that mismatch hid a real bug (P/T axes computed in the wrong precision) for a while.
    """
    if got == ref:
        return True

    def skeleton(t):
        return _NUM.sub(lambda m: "#" * len(m.group()), t)

    if skeleton(got) != skeleton(ref):
        return False
    g = [float(x) for x in _NUM.findall(got)]
    r = [float(x) for x in _NUM.findall(ref)]
    return all(abs(a - b) <= tol for a, b in zip(g, r))


def compare_files(got_path, ref_path, skip_first=True, tol=0.0105, noise=()):
    """Byte-exact comparison of two listings.

    Only two kinds of difference are tolerated: lines whose index is in ``noise`` (a last printed
    digit that differs because numpy and libm round float32 trig differently) and the documented
    P/T axis divergence.  Returns the number of lines that are not identical.
    """
    got = open(got_path).read().splitlines()
    ref = open(ref_path).read().splitlines()
    if skip_first:  # time stamp
        got, ref = got[1:], ref[1:]
    assert len(got) == len(ref), f"{got_path}: {len(got)} lines vs reference {len(ref)}"
    bad = [
        (i, g, r)
        for i, (g, r) in enumerate(zip(got, ref))
        if g != r and not ((i in noise and lines_match(g, r, tol)) or pt_axis_divergence(g, r, tol))
    ]
    assert not bad, f"{len(bad)} differing lines; first: {bad[0]}"
    return sum(g != r for g, r in zip(got, ref))
