"""Plot symbols after the centred special symbols of the original (``SPCSMB``, see ``symdemo.pdf``).

``SHAPES[n]`` is a list of pen strokes, each a list of ``(x, y)`` points in a box of +-10 about the
centre.  The numbers follow the original: 0 box with tick (SV/P ratio), 1 octagon (P compression, SV/SH
ratio), 2 triangle (P dilatation), 3 plus (emergent compression, centre mark), 5 diamond (SH/P ratio),
6 arrow (SV and SH polarities, pointing up), 12 hourglass (ratio reference size).
"""

from __future__ import annotations

import numpy as np

_TICK = [(0, 0), (0, 10)]

SHAPES: dict[int, list[list[tuple[float, float]]]] = {
    0: [[(-10, -10), (10, -10), (10, 10), (-10, 10), (-10, -10)], _TICK],
    1: [[(4, 10), (10, 4), (10, -4), (4, -10), (-4, -10), (-10, -4), (-10, 4), (-4, 10), (4, 10)], _TICK],
    2: [[(0, 10), (9, -5), (-9, -5), (0, 10)], _TICK],
    3: [[(0, -10), (0, 10)], [(-10, 0), (10, 0)]],
    5: [[(0, 10), (10, 0), (0, -10), (-10, 0), (0, 10)], _TICK],
    6: [[(0, -10), (0, 10)], [(0, 10), (10, 0), (-10, 0), (0, 10)]],
    12: [[(-10, 10), (10, 10), (-10, -10), (10, -10), (-10, 10)]],
}


def strokes(ns: int, height: float, angle: float = 0.0, x: float = 0.0, y: float = 0.0):
    """Strokes of symbol ``ns`` as arrays of ``(x, y)``: ``height`` high (the box is 20/21 of it), rotated
    ``angle`` degrees counter-clockwise about the centre, and moved to ``(x, y)``."""
    t = np.radians(angle)
    c, s = np.cos(t), np.sin(t)
    out = []
    for stroke in SHAPES[ns]:
        p = np.asarray(stroke, dtype=float) * (height / 21.0)
        out.append(np.column_stack((x + p[:, 0] * c - p[:, 1] * s, y + p[:, 0] * s + p[:, 1] * c)))
    return out


def draw_symbol(ax, x, y, ns, height, angle=0.0, lw=1.0, color="k", zorder=3):
    """Draw symbol ``ns`` centred on ``(x, y)`` in data units."""
    for p in strokes(ns, height, angle, x, y):
        ax.plot(p[:, 0], p[:, 1], "-", color=color, lw=lw, solid_joinstyle="miter", solid_capstyle="butt",
                zorder=zorder)
