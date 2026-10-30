"""Focal-sphere plots (matplotlib replacements for ``focplt`` and ``ratplt``).

All plots use the lower-hemisphere equal-area (Schmidt) projection of the
original, with north up; ``upper=True`` gives the upper hemisphere.  Needs
``matplotlib`` (``pip install pyfocmec[plot]``).

Symbol sizes, dash lengths and label positions are given in the inches of the original programs, whose
focal sphere has a radius of 2.3 in (``size=0.25`` is what ``focplt`` offered as its default size);
they are scaled with the sphere ``radius``.  Line widths are pen numbers as in the original.
"""

from __future__ import annotations

import numpy as np

from . import dsretc
from ._symbols import draw_symbol
from .geometry import dsr_to_an
from .inp import parse_inp, read_picks
from .radiation import FLAG_ND, log_ratio

_R0 = 2.3  # radius of the focal sphere in the original programs, inches
_PEN_PT = 0.75  # points per pen number
_FIG_IN = 4.2  # default figure size, inches


def _mpl():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plotting needs matplotlib: pip install pyfocmec[plot]") from exc
    return plt


def _u(radius):
    """Data units per inch of the original plots."""
    return radius / _R0


def _pen(n):
    return _PEN_PT * float(n)


def _pts_per_unit(ax):
    """Points per data unit of ``ax`` (for sizing text and dashes)."""
    ax.apply_aspect()
    x0, x1 = ax.get_xlim()
    return ax.get_window_extent().width / abs(x1 - x0) * 72.0 / ax.figure.dpi


def _text(ax, x, y, s, height, radius, color="k", ha="left", va="baseline"):
    """Text whose capitals are about ``height`` (original inches) high."""
    size = height * _u(radius) * _pts_per_unit(ax) / 0.72
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va)


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
def focal_sphere(ax=None, radius=1.0, upper=False, title=None, caption=False, label=None, label_xy=(-2.0, 2.2)):
    """Draw the empty focal sphere (circle, centre mark, ticks and N, as ``cirplt``) on ``ax``.

    ``title`` goes under the circle.  ``label`` (for example ``"a)"``) is placed at ``label_xy``, in
    inches from the centre of the original plot.  ``caption`` adds a "lower/upper hemisphere" note.
    """
    plt = _mpl()
    lim = radius * 1.18
    if ax is None:
        fig = plt.figure(figsize=(_FIG_IN, _FIG_IN))
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
        if title:
            ax.set_ylim(-radius * 1.42, lim)
            fig.set_size_inches(_FIG_IN, _FIG_IN * (lim + radius * 1.42) / (2 * lim))
        else:
            ax.set_ylim(-lim, lim)
        ax.set_xlim(-lim, lim)
    else:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-radius * 1.42 if title else -lim, lim)
    u = _u(radius)
    th = np.linspace(0, 2 * np.pi, 361)
    ax.plot(radius * np.cos(th), radius * np.sin(th), "k-", lw=_pen(2), zorder=3)
    draw_symbol(ax, 0.0, 0.0, 3, 0.1 * u, lw=_pen(1))
    tick = 0.2 * u
    for x0, y0, x1, y1 in ((0, radius - tick, 0, radius), (0, -radius + tick, 0, -radius),
                           (radius - tick, 0, radius, 0), (-radius + tick, 0, -radius, 0)):
        ax.plot([x0, x1], [y0, y1], "k-", lw=_pen(2), solid_capstyle="butt", zorder=3)
    _text(ax, 0.0, radius + 0.1 * u, "N", 0.25, radius, ha="center")
    if caption:
        ax.text(0, -radius - 0.12 * u, "upper hemisphere" if upper else "lower hemisphere", ha="center", va="top",
                fontsize=8, color="0.4")
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        _text(ax, 0.0, -radius - 0.6 * u, title, 0.2, radius, ha="center")
    if label:
        _text(ax, label_xy[0] * u, label_xy[1] * u, label, 0.2, radius)
    return ax


_PHASE_OF = {"C": "P", "D": "P", "+": "P", "-": "P", "e": "P", "<": "SH", ">": "SH", "l": "SH", "r": "SH",
             "u": "SH", "F": "SV", "B": "SV"}
_ARROW_ANGLE = {"<": 450.0, ">": 270.0, "F": 360.0, "B": 180.0}  # minus the azimuth, as in polplot.f


def plot_polarities(picks, ax=None, radius=1.0, upper=False, phases=("P", "SV", "SH"), emergent=False,
                    size=0.25, emergent_size=0.15, sh_size=None, sv_size=None, lw=1.0, errors=False, color="k",
                    labels=False):
    """Plot polarity picks with the symbols of ``focplt``.

    ``picks`` is a list of :class:`pyfocmec.inp.Pick` or a path to a ``.inp`` file (a parsed
    ``FocmecData`` has no raw picks and is not accepted).  Compressions (``C``, ``U``) are octagons and
    dilatations (``D``) triangles, both ``size`` high; SH (``< >``) and SV (``F B``) first motions are
    arrows of ``sh_size``/``sv_size`` (default ``size``) pointing the way the ground moves.  With
    ``emergent`` the emergent picks (``+ - e l r u``) are drawn at ``emergent_size``.  With ``errors``
    an ``E`` line flags the preceding pick with a box of its size.  Sizes are in the inches of the
    original ``focplt`` (see the module docstring) and ``lw`` is a pen number.
    """
    if isinstance(picks, str) or hasattr(picks, "__fspath__"):
        picks = read_picks(picks)[1]
    elif not isinstance(picks, (list, tuple)):
        raise TypeError("picks must be a path to a .inp file or a list of Pick objects")
    if ax is None:
        ax = focal_sphere(radius=radius, upper=upper)
    u = _u(radius)
    sizes = {"P": size, "SH": size if sh_size is None else sh_size, "SV": size if sv_size is None else sv_size}
    last = None
    for p in picks:
        code = {"U": "C", "R": ">", "L": "<"}.get(p.code, p.code)
        if p.is_ratio:
            continue
        if code == "E":
            if errors and last is not None:
                x, y = (float(v) for v in project(p.azimuth, p.takeoff, upper, radius))
                draw_symbol(ax, x, y, 0, last * u, lw=_pen(lw), color=color)
            continue
        if code not in _PHASE_OF or _PHASE_OF[code] not in phases:
            continue
        impulsive = code in "CD<>FB"
        if not impulsive and not emergent:
            continue
        h = sizes[_PHASE_OF[code]] if impulsive else emergent_size
        x, y = (float(v) for v in project(p.azimuth, p.takeoff, upper, radius))
        if code == "-":
            ax.plot([x - 0.3 * h * u, x + 0.3 * h * u], [y, y], "-", color=color, lw=_pen(lw))
        elif code in "elru":
            _text(ax, x, y, code, h, radius, color=color, ha="center", va="center")
        elif code in "CD":
            draw_symbol(ax, x, y, 1 if code == "C" else 2, h * u, lw=_pen(lw), color=color)
        elif code == "+":
            draw_symbol(ax, x, y, 3, h * u, lw=_pen(lw), color=color)
        else:
            az = p.azimuth + (180.0 if p.takeoff > 90 else 0.0) + (180.0 if upper else 0.0)
            draw_symbol(ax, x, y, 6, h * u, angle=_ARROW_ANGLE[code] - az, lw=_pen(lw), color=color)
        last = h
        if labels:
            ax.text(x, y, " " + p.sta, fontsize=7, color="0.35", va="bottom")
    return ax


def _dash_style(ax, radius, dash, lw):
    """matplotlib linestyle for a dash ``(length, gap)`` in inches of the original plot (None: solid)."""
    if dash is None:
        return "-"
    k = _u(radius) * _pts_per_unit(ax) / _pen(lw)  # dash patterns scale with the line width
    return (0, (dash[0] * k, dash[1] * k))


def plot_solution(dsr, ax=None, radius=1.0, upper=False, color="k", lw=1.0, axes=(), axes_size=0.15,
                  surface="P", dash=None, label=None, alpha=1.0):
    """Draw one mechanism ``(dip, strike, rake)`` as ``solplt`` does.

    ``surface`` picks what is drawn: the two fault planes (``"P"``) or the nodal surface of SH or SV
    waves.  ``axes`` (any of ``"P"``, ``"T"``, ``"B"``) adds the letters of those axes, ``dash`` is
    ``(length, gap)`` in the inches of the original plot, ``lw`` a pen number.
    """
    if ax is None:
        ax = focal_sphere(radius=radius, upper=upper)
    ls = _dash_style(ax, radius, dash, lw)
    if surface == "P":
        rep = dsretc.from_dsr(*dsr)
        for plane in (rep.dsr, rep.aux):
            x, y = nodal_plane_xy(plane[0], plane[1], upper, radius)
            ax.plot(x, y, ls=ls, color=color, lw=_pen(lw), alpha=alpha, label=label, zorder=2)
            label = None
    elif surface in ("SH", "SV"):
        X, Y, F = radiation_on_sphere(dsr, surface, upper=upper, radius=radius)
        ax.contour(X, Y, F, levels=[0.0], colors=[color], linestyles=[ls], linewidths=_pen(lw), alpha=alpha,
                   zorder=2)
    else:
        raise ValueError("surface must be P, SH or SV")
    if axes:
        rep = dsretc.from_dsr(*dsr)
        marks = {"P": rep.p, "T": rep.t, "B": rep.b}
        for k in axes:
            x, y = axis_xy(*marks[k], upper=upper, radius=radius)
            _text(ax, float(x), float(y), k, axes_size, radius, color=color, ha="center", va="center")
    return ax


def _rows(solutions):
    """``(dip, strike, rake)`` rows from a ``FocmecResult``, its solutions, a ``focmec.out`` path or rows."""
    if hasattr(solutions, "solutions"):
        solutions = solutions.solutions
    if isinstance(solutions, str) or hasattr(solutions, "__fspath__"):
        return [(r["dip"], r["strike"], r["rake"]) for r in read_out(solutions)]
    return [(s.dip, s.strike, s.rake) if hasattr(s, "dip") else tuple(s) for s in solutions]


def plot_solutions(solutions, ax=None, radius=1.0, upper=False, color="k", alpha=1.0, axes=(), lw=1.0,
                   surface="P", dash=None, axes_size=0.15):
    """Overlay many solutions (a ``SolutionSet``, list of ``Solution``, ``focmec.out`` path or
    ``(dip, strike, rake)`` rows), all in one style (see :func:`plot_solution`)."""
    if ax is None:
        ax = focal_sphere(radius=radius, upper=upper)
    for dsr in _rows(solutions):
        plot_solution(dsr, ax=ax, radius=radius, upper=upper, color=color, lw=lw, axes=axes, alpha=alpha,
                      surface=surface, dash=dash, axes_size=axes_size)
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


def plot_nodal_surfaces(dsr, ax=None, radius=1.0, upper=False, kinds=("SV", "SH"), colors=None, lw=1.0, dash=None):
    """Zero-radiation (nodal) surfaces of SV and/or SH waves for a mechanism."""
    if ax is None:
        ax = focal_sphere(radius=radius, upper=upper)
    colors = colors or {}
    for k in kinds:
        plot_solution(dsr, ax=ax, radius=radius, upper=upper, color=colors.get(k, "k"), lw=lw, surface=k,
                      dash=dash if dash is not None else ((0.15, 0.05) if k == "SV" else (0.02, 0.05)))
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
           nodal_surfaces=False, beachball=False, axes=("P", "T"), title=None, ax=None, labels=False, sets=None,
           label=None, size=0.25, emergent_size=0.15, sh_size=None, sv_size=None, errors=False, caption=False):
    """Data and/or solutions on the focal sphere (what ``focplt`` does).

    ``inp``: path to a FOCMEC input file (polarities are plotted; ``size``, ``emergent_size``,
    ``sh_size``, ``sv_size`` and ``errors`` as in :func:`plot_polarities`).

    Solutions are drawn as *sets*, each with its own style, exactly as ``focplt`` asks for them one
    after the other.  ``sets`` is a list of dicts with keys ``solutions`` (a :class:`pyfocmec.FocmecResult`,
    its ``solutions``, a ``focmec.out`` path or rows of ``(dip, strike, rake)``) or ``dsr`` (one
    mechanism), and optionally ``surface`` (``"P"`` planes, ``"SH"`` or ``"SV"`` nodal surfaces),
    ``dash`` (``(length, gap)``, inches), ``lw`` (pen), ``axes`` (letters of the P/T/B axes to mark)
    and ``color``.

    ``solutions`` and ``dsr`` are shorthand for a plain set of solutions and one mechanism marked with
    its ``axes`` (and with its SV/SH nodal surfaces when ``nodal_surfaces``); ``beachball`` also shades
    the quadrants of ``dsr``.  ``label`` is a plot label such as ``"a)"``.  Returns the matplotlib ``Axes``.
    """
    ax = focal_sphere(ax=ax, upper=upper, title=title, caption=caption, label=label)
    if dsr is not None and beachball:
        plot_beachball(dsr, ax=ax, upper=upper)
    if inp is not None:
        plot_polarities(inp, ax=ax, upper=upper, phases=phases, emergent=emergent, labels=labels, size=size,
                        emergent_size=emergent_size, sh_size=sh_size, sv_size=sv_size, errors=errors)
    sets = list(sets or [])
    if solutions is not None:
        sets.append({"solutions": solutions})
    if dsr is not None:
        sets.append({"dsr": dsr, "lw": 1.5, "axes": tuple(axes)})
        if nodal_surfaces:
            sets += [{"dsr": dsr, "surface": "SV", "dash": (0.15, 0.05)},
                     {"dsr": dsr, "surface": "SH", "dash": (0.02, 0.05)}]
    for s in sets:
        s = dict(s)
        rows = _rows(s.pop("solutions")) if "solutions" in s else [tuple(s.pop("dsr"))]
        plot_solutions(rows, ax=ax, upper=upper, **s)
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


_RATIO_SYMBOL = {"V": 0, "H": 5, "S": 1}  # SV/P box, SH/P diamond, SV/SH octagon, as ratplt.f


def ratplt(inp, dsr, error=0.1, cutp=0.1, cuts=0.1, scale=0.20, upper=False, title=None, ax=None, label=None,
           legend=False):
    """Observed-minus-calculated log amplitude ratios on the focal sphere (``ratplt``).

    Each ratio is drawn at its station position with a symbol of height ``scale * 10**|difference|``
    inches of the original plot (box = SV/P, diamond = SH/P, octagon = SV/SH), and every station gets an
    hourglass of the reference height ``scale * 10**error``.  Ratios with both parts near a nodal
    surface are skipped.  ``title`` goes under the circle, ``label`` is a plot label such as ``"a)"``,
    and ``legend`` adds a key to the symbols.
    """
    _mpl()
    _, picks = read_picks(inp)
    ratios = [p for p in picks if p.is_ratio]
    if not ratios:
        raise ValueError("no amplitude ratios in input")
    data = parse_inp(inp)
    vpvs3 = data.vpvs**3
    dsr_r = np.radians(np.asarray(dsr, dtype=float))
    ax = focal_sphere(ax=ax, upper=upper, title=title, label=label)
    u = _u(1.0)
    jr = {"V": 1, "H": 2, "S": 3}
    for k, p in enumerate(ratios):
        lr, _, flag = log_ratio(jr[p.code], dsr_r[0], dsr_r[1], dsr_r[2], data.rat_xyz[k], data.rat_xyzden[k],
                                vpvs3, cutp, cuts)
        if int(flag) == FLAG_ND:
            continue
        diff = p.log_ratio - float(lr)
        x, y = (float(v) for v in project(p.azimuth, p.takeoff, upper))
        draw_symbol(ax, x, y, _RATIO_SYMBOL[p.code], scale * 10 ** abs(diff) * u, lw=_pen(1))
        draw_symbol(ax, x, y, 12, scale * 10**error * u, lw=_pen(1))
    if legend:
        from matplotlib.lines import Line2D

        handles = [Line2D([], [], marker=m, ls="", mfc="none", mec="k", label=l)
                   for m, l in (("s", "SV/P"), ("D", "SH/P"), ("8", "SV/SH"))]
        ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=False, bbox_to_anchor=(-0.1, -0.08))
    return ax
