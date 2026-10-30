"""Tiny helpers that reproduce the Fortran edit descriptors FOCMEC output relies on."""

from __future__ import annotations

import numpy as np


def fnum(x: float, w: int, d: int) -> str:
    """Fortran ``Fw.d`` (right-justified; asterisks on overflow)."""
    # the Fortran program holds these values as REAL*4, which decides rounding
    v = float(np.float32(x))
    if v != v:  # gfortran: NaN, right-justified
        return "NaN".rjust(w) if w >= 3 else "*" * w
    if v in (float("inf"), float("-inf")):  # gfortran: Infinity, or Inf when the field is narrow
        sign = "-" if v < 0 else ""
        word = "Infinity" if w >= len(sign) + 8 else "Inf"
        return (sign + word).rjust(w) if len(sign + word) <= w else "*" * w
    s = f"{v:.{d}f}"
    if len(s) > w:
        # Fortran drops the leading zero before overflowing
        if s.startswith("0."):
            s = s[1:]
        elif s.startswith("-0."):
            s = "-" + s[2:]
    return s.rjust(w) if len(s) <= w else "*" * w


def inum(i: int, w: int, zeros: bool = False) -> str:
    """Fortran ``Iw`` or ``Iw.w`` (zero padded)."""
    s = f"{int(i):0{w}d}" if zeros else f"{int(i):d}"
    return s.rjust(w) if len(s) <= w else "*" * w


def rec(*parts) -> str:
    """Build one output record.

    Integers are Fortran ``Tn`` tab positions (1-based column of the next
    character); strings are written at the current position, overwriting or
    extending the record.
    """
    buf: list[str] = []
    pos = 0
    for p in parts:
        if isinstance(p, int):
            pos = p - 1
            continue
        if pos > len(buf):
            buf.extend(" " * (pos - len(buf)))
        for ch in p:
            if pos < len(buf):
                buf[pos] = ch
            else:
                buf.append(ch)
            pos += 1
    return "".join(buf)
