"""Focal-sphere plots (matplotlib replacements for ``focplt`` and ``ratplt``).

All plots use the lower-hemisphere equal-area (Schmidt) projection of the
original, with north up; ``upper=True`` gives the upper hemisphere.  Needs
``matplotlib`` (``pip install pyfocmec[plot]``).
"""

from __future__ import annotations

import numpy as np

from . import dsretc
from .geometry import dsr_to_an
from .inp import read_picks
from .radiation import FLAG_ND, log_ratio


def _mpl():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plotting needs matplotlib: pip install pyfocmec[plot]") from exc
    return plt


# ---------------------------------------------------------------- projection
def project(azimuth, takeoff, upper=False, radius=1.0):
    """Equal-area projection of rays given azimuth and take-off angle (degrees from down).

    Upgoing rays (take-off > 90) are plotted at their antipode, as is customary.
    Returns ``(x, y)`` with x east and y north.
    """
    az = np.asarray(azimuth, dtype=float).copy()
    toa = np.asarray(takeoff, dtype=float).copy()
    up = toa > 90.0
    toa = np.where(up, 180.0 - toa, toa)
    az = np.where(up, az + 180.0, az)
    if upper:
        az = az + 180.0
    r = radius * np.sqrt(2.0) * np.sin(np.radians(toa) / 2.0)
    return r * np.sin(np.radians(az)), r * np.cos(np.radians(az))


def axis_xy(trend, plunge, upper=False, radius=1.0):
    """Projected position of an axis given by trend and plunge (degrees)."""
    return project(trend, 90.0 - np.asarray(plunge, dtype=float), upper, radius)


def nodal_plane_xy(dip, strike, upper=False, radius=1.0, n=181):
    """Projected great circle of a fault plane (lower-hemisphere half)."""
    d, s = np.radians(dip), np.radians(strike)
    nvec = np.array([-np.sin(s) * np.sin(d), np.cos(s) * np.sin(d), -np.cos(d)])
    u = np.array([np.cos(s), np.sin(s), 0.0])
    w = -np.cross(nvec, u)
    phi = np.linspace(0.0, np.pi, n)
    pts = np.outer(np.cos(phi), u) + np.outer(np.sin(phi), w)
    z = np.clip(pts[:, 2], 0.0, 1.0)
    toa = np.degrees(np.arccos(z))
    az = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    return project(az, toa, upper, radius)


# ---------------------------------------------------------------- drawing
def focal_sphere(ax=None, radius=1.0, upper=False, title=None):
    """Draw the empty focal sphere (circle, centre cross, N tick) on ``ax``."""
    plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6.4))
    th = np.linspace(0, 2 * np.pi, 361)
    ax.plot(radius * np.cos(th), radius * np.sin(th), "k-", lw=1.6)
    ax.plot([0], [0], "k+", ms=6)
    for (x0, y0, x1, y1) in ((0, radius - 0.08 * radius, 0, radius), (0, -radius + 0.08 * radius, 0, -radius),
                             (radius - 0.08 * radius, 0, radius, 0), (-radius + 0.08 * radius, 0, -radius, 0)):
        ax.plot([x0, x1], [y0, y1], "k-", lw=1.2)
    ax.text(0, radius * 1.04, "N", ha="center", va="bottom", fontsize=12)
    ax.text(0, -radius * 1.04, "upper hemisphere" if upper else "lower hemisphere", ha="center", va="top",
            fontsize=8, color="0.4")
    ax.set_aspect("equal")
    lim = radius * 1.18
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)
    return ax


_PHASE_OF = {"C": "P", "D": "P", "+": "P", "-": "P", "e": "P", "<": "SH", ">": "SH", "l": "SH", "r": "SH",
             "u": "SH", "F": "SV", "B": "SV"}


def plot_polarities(picks, ax=None, radius=1.0, upper=False, phases=("P", "SV", "SH"), emergent=False,
                    size=0.07, color="k", labels=False):
    """Plot polarity picks.

    ``picks`` is a list of :class:`pyfocmec.inp.Pick` or a path to a ``.inp`` file (a parsed
    ``FocmecData`` has no raw picks and is not accepted).
    Symbols: filled circle = P compression (up), open circle = P dilatation;
    arrows: SV forward (away from the centre) / backward, SH left (counter-clockwise) / right.
    Emergent picks (``+ - l r e u``) are drawn as small letters when ``emergent`` is true.
    """
    if isinstance(picks, str) or hasattr(picks, "__fspath__"):
        picks = read_picks(picks)[1]
    elif not isinstance(picks, (list, tuple)):
        raise TypeError("picks must be a path to a .inp file or a list of Pick objects")
    if ax is None:
        ax = focal_sphere(radius=radius, upper=upper)
    size = size * radius
    for p in picks:
        code = {"U": "C", "R": ">", "L": "<"}.get(p.code, p.code)
        if p.is_ratio or code not in _PHASE_OF or _PHASE_OF[code] not in phases:
            continue
        x, y = project(p.azimuth, p.takeoff, upper, radius)
        x, y = float(x), float(y)
        if code in "CD":
            ax.plot([x], [y], "o", ms=size * 70, mfc=color if code == "C" else "white", mec=color, mew=1.2)
        elif code in "+-eulr":
            if emergent:
                ax.text(x, y, {"+": "+", "-": "–", "e": "e", "u": "u", "l": "l", "r": "r"}[code], ha="center",
                        va="center", fontsize=9, color=color)
        else:
            az = (p.azimuth + (180.0 if p.takeoff > 90 else 0.0) + (180.0 if upper else 0.0)) % 360.0
            azr = np.radians(az)
            radial = np.array([np.sin(azr), np.cos(azr)])
            ccw = np.array([-np.cos(azr), np.sin(azr)])
            d = {"F": radial, "B": -radial, "<": ccw, ">": -ccw}[code]
            ax.annotate("", xy=(x + d[0] * size * 1.6, y + d[1] * size * 1.6),
                        xytext=(x - d[0] * size * 1.6, y - d[1] * size * 1.6),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, mutation_scale=10))
        if labels:
            ax.text(x, y, " " + p.sta, fontsize=7, color="0.35", va="bottom")
    return ax


def plot_solution(dsr, ax=None, radius=1.0, upper=False, color="tab:red", lw=1.5, axes=("P", "T"),
                  label=None, alpha=1.0):
    """Draw the two nodal planes (and optionally P/T/B axes) of a mechanism ``(dip, strike, rake)``."""
    if ax is None:
        ax = focal_sphere(radius=radius, upper=upper)
    rep = dsretc.from_dsr(*dsr)
    for plane in (rep.dsr, rep.aux):
        x, y = nodal_plane_xy(plane[0], plane[1], upper, radius)
        ax.plot(x, y, "-", color=color, lw=lw, alpha=alpha, label=label)
        label = None
    marks = {"P": rep.p, "T": rep.t, "B": rep.b}
    for k in axes:
        x, y = axis_xy(*marks[k], upper=upper, radius=radius)
        ax.text(float(x), float(y), k, ha="center", va="center", fontsize=11, fontweight="bold", color=color)
    return ax


def plot_solutions(solutions, ax=None, radius=1.0, upper=False, color="tab:red", alpha=None, axes=(), lw=1.0):
    """Overlay many solutions (a ``SolutionSet``, list of ``Solution`` or ``(dip, strike, rake)`` rows)."""
    if ax is None:
        ax = focal_sphere(radius=radius, upper=upper)
    rows = [(s.dip, s.strike, s.rake) if hasattr(s, "dip") else tuple(s) for s in solutions]
    if alpha is None:
        alpha = min(1.0, max(0.25, 8.0 / max(len(rows), 1)))
    for dsr in rows:
        plot_solution(dsr, ax=ax, radius=radius, upper=upper, color=color, lw=lw, axes=axes, alpha=alpha)
    return ax


def _disk_grid(radius, n=241):
    xs = np.linspace(-radius, radius, n)
    X, Y = np.meshgrid(xs, xs)
    rr = np.hypot(X, Y)
    inside = rr <= radius
    sin_half = np.clip(rr / (radius * np.sqrt(2.0)), 0.0, 1.0)
    toa = 2.0 * np.arcsin(sin_half)
    az = np.arctan2(X, Y)  # clockwise from north
    return X, Y, inside, toa, az


def radiation_on_sphere(dsr, kind, n=241, upper=False, radius=1.0):
    """Radiation factor of ``kind`` ("P", "SV" or "SH") on the projected disk (lower-hemisphere rays)."""
    X, Y, inside, toa, az = _disk_grid(radius, n)
    if upper:  # same pattern seen from above: antipodal rays
        az = az + np.pi
    a, nvec = dsr_to_an(*(np.radians(v) for v in dsr))
    r = np.stack([np.sin(toa) * np.cos(az), np.sin(toa) * np.sin(az), np.cos(toa)], -1)
    ra, rn = r @ a, r @ nvec
    if kind == "P":
        f = 2.0 * ra * rn
    elif kind == "SV":
        th = np.stack([-np.cos(toa) * np.cos(az), -np.cos(toa) * np.sin(az), np.sin(toa)], -1)
        f = rn * (th @ a) + ra * (th @ nvec)
    elif kind == "SH":
        ph = np.stack([np.sin(az), -np.cos(az), np.zeros_like(az)], -1)
        f = rn * (ph @ a) + ra * (ph @ nvec)
    else:
        raise ValueError("kind must be P, SV or SH")
    return X, Y, np.where(inside, f, np.nan)


def plot_nodal_surfaces(dsr, ax=None, radius=1.0, upper=False, kinds=("SV", "SH"), colors=None):
    """Zero-radiation (nodal) surfaces of SV and/or SH waves for a mechanism."""
    if ax is None:
        ax = focal_sphere(radius=radius, upper=upper)
    colors = colors or {"SV": "tab:blue", "SH": "tab:green", "P": "tab:red"}
    styles = {"SV": "--", "SH": ":", "P": "-"}
    for k in kinds:
        X, Y, F = radiation_on_sphere(dsr, k, upper=upper, radius=radius)
        ax.contour(X, Y, F, levels=[0.0], colors=[colors[k]], linestyles=styles[k], linewidths=1.2)
    return ax


def plot_beachball(dsr, ax=None, radius=1.0, upper=False, compression="0.25"):
    """Shade the compressional quadrants of a mechanism."""
    if ax is None:
        ax = focal_sphere(radius=radius, upper=upper)
    X, Y, F = radiation_on_sphere(dsr, "P", upper=upper, radius=radius)
    ax.contourf(X, Y, F, levels=[0.0, np.inf], colors=[compression], zorder=0)
    return ax


# ---------------------------------------------------------------- high level
def focplt(inp=None, solutions=None, dsr=None, upper=False, phases=("P", "SV", "SH"), emergent=False,
           nodal_surfaces=False, beachball=False, axes=("P", "T"), title=None, ax=None, labels=False):
    """Data and/or solutions on the focal sphere (what ``focplt`` does).

    ``inp``: path to a FOCMEC input file (polarities are plotted).  ``solutions``:
    a :class:`pyfocmec.FocmecResult`, its ``solutions``, a ``focmec.out`` path or rows of
    ``(dip, strike, rake)``.  ``dsr``: one mechanism drawn emphasised (with axes, and
    optionally SV/SH nodal surfaces).  Returns the matplotlib ``Axes``.
    """
    ax = focal_sphere(ax=ax, upper=upper, title=title)
    if dsr is not None and beachball:
        plot_beachball(dsr, ax=ax, upper=upper)
    if inp is not None:
        plot_polarities(inp, ax=ax, upper=upper, phases=phases, emergent=emergent, labels=labels)
    if solutions is not None:
        if hasattr(solutions, "solutions"):
            solutions = solutions.solutions
        if isinstance(solutions, (str,)) or hasattr(solutions, "__fspath__"):
            solutions = [(r["dip"], r["strike"], r["rake"]) for r in read_out(solutions)]
        plot_solutions(solutions, ax=ax, upper=upper)
    if dsr is not None:
        plot_solution(dsr, ax=ax, upper=upper, color="tab:blue", lw=2.0, axes=axes)
        if nodal_surfaces:
            plot_nodal_surfaces(dsr, ax=ax, upper=upper)
    return ax


def read_out(path) -> list[dict]:
    """Solutions in a ``focmec.out`` file: dip, strike and rake, plus the misfit columns when present.

    Like ``solplt.f``, everything up to the line starting with ``    Dip`` is header; after it
    each line gives dip, strike and rake of a solution.
    """
    rows = []
    in_header = True
    with open(path, errors="replace") as fh:
        for line in fh:
            if in_header:
                in_header = not line.startswith("    Dip")
                continue
            parts = line.split()
            try:
                vals = [float(x) for x in parts[:3]]
            except ValueError:
                continue
            if len(vals) < 3:
                continue
            row = dict(dip=vals[0], strike=vals[1], rake=vals[2])
            try:
                row.update(bad_p=float(parts[3]), bad_sv=float(parts[4]), bad_sh=float(parts[5]))
            except (IndexError, ValueError):
                pass
            if len(parts) >= 9 and "/" in parts[6]:
                ok, tot = parts[6].split("/")
                row.update(ratio_ok=int(ok), ratio_total=int(tot), ratio_rms=float(parts[7]),
                           ratio_maxdiff=float(parts[8]))
            rows.append(row)
    if in_header:
        raise ValueError(f"{path}: no line starting with '    Dip' (not a focmec.out file?)")
    return rows


def ratplt(inp, dsr, error=0.1, cutp=0.1, cuts=0.1, scale=0.20, upper=False, title=None, ax=None):
    """Observed-minus-calculated log amplitude ratios on the focal sphere (``ratplt``).

    Each ratio is drawn at its station position with a symbol whose size is
    ``scale * 10**|difference|`` (square = SV/P, triangle = SH/P, circle = SV/SH).
    The dashed circle is the reference size for ``error``.  Ratios with both parts
    near a nodal surface are skipped.
    """
    _mpl()
    _, picks = read_picks(inp)
    ratios = [p for p in picks if p.is_ratio]
    if not ratios:
        raise ValueError("no amplitude ratios in input")
    from .inp import parse_inp

    data = parse_inp(inp)
    vpvs3 = data.vpvs**3
    dsr_r = np.radians(np.asarray(dsr, dtype=float))
    ax = focal_sphere(ax=ax, upper=upper, title=title or f"Ratio misfit for dip/strike/rake {dsr[0]:.0f}/{dsr[1]:.0f}/{dsr[2]:.0f}")
    jr = {"V": 1, "H": 2, "S": 3}
    for k, p in enumerate(ratios):
        lr, _, flag = log_ratio(jr[p.code], dsr_r[0], dsr_r[1], dsr_r[2], data.rat_xyz[k], data.rat_xyzden[k],
                                vpvs3, cutp, cuts)
        if int(flag) == FLAG_ND:
            continue
        diff = p.log_ratio - float(lr)
        x, y = project(p.azimuth, p.takeoff, upper)
        marker = {"V": "s", "H": "^", "S": "o"}[p.code]
        # scatter(s=) is an area: the symbol's linear size is scale * 10**|diff| as in ratplt.f
        ax.scatter([x], [y], s=960 * scale * (10 ** abs(diff)) ** 2, marker=marker, facecolors="none",
                   edgecolors="k", lw=1.2)
        ax.scatter([x], [y], s=960 * scale * (10**error) ** 2, marker="o", facecolors="none",
                   edgecolors="0.5", lw=0.6, linestyles="--")
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker=m, ls="", mfc="none", mec="k", label=l)
               for m, l in (("s", "SV/P"), ("^", "SH/P"), ("o", "SV/SH"))]
    handles.append(Line2D([], [], marker="o", ls="", mfc="none", mec="0.5", label=f"|o-c| = {error:g}"))
    ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=False, bbox_to_anchor=(-0.1, -0.08))
    return ax
