"""Reader/writer for the FOCMEC ``.inp`` input format (port of ``focinp.f``).

Line 1 of the file is a comment.  Every following line is either a polarity

    ``A4,2F8.2,A1``  station, azimuth, take-off angle, key  (comment from col 40)

or an amplitude ratio (recognised by non-blank columns 24-29)

    ``A4,2F8.2,A1,F8.4,1X,A1,1X,F6.2,1X,F6.4``
    station, azimuth, numerator take-off angle, key (V = SV/P, H = SH/P,
    S = SV/SH), log10 ratio, numerator polarity, denominator take-off angle,
    focus Vp/Vs                                    (comment from col 46)
"""

from __future__ import annotations

import io
import os
import warnings
from dataclasses import dataclass, field

import numpy as np

from .fortfmt import fnum, rec

# Keys that denote emergent picks.  Deliberate divergence from the original: focinp.f
# converts emergent 'f'/'b' (SV) picks but the test that lets them reach that code only
# admits '+', '-', 'l' and 'r', so in the original they were silently dropped.
EMERGENT_KEYS = "+-lrfb"
_EMERGENT_MAP = {"+": "C", "-": "D", "l": "<", "r": ">", "f": "F", "b": "B"}


class InputError(ValueError):
    """Malformed FOCMEC input line."""
_P_UP, _P_DOWN = "C", "D"


@dataclass
class FocmecData:
    """Parsed FOCMEC input, with the direction vectors used by the search."""

    comment: str = ""
    filename: str = ""
    echo: list[str] = field(default_factory=list)  # lines echoed to the .lst file
    emergent_msg_index: int | None = None  # echo index after which the emergent message goes
    has_emergent: bool = False
    include_emergent: bool | None = None
    # polarities
    pol_sta: list[str] = field(default_factory=list)
    pol_sign: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pol_kind: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))  # 0 P, 1 SV, 2 SH
    pol_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 9)))
    pol_xyz32: np.ndarray = field(default_factory=lambda: np.zeros((0, 9), dtype=np.float32))
    # ratios
    rat_sta: list[str] = field(default_factory=list)
    rat_sense: list[str] = field(default_factory=list)  # V, H or S
    rat_numpol: list[str] = field(default_factory=list)
    rat_log: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rat_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 9)))
    rat_xyzden: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    # the same vectors evaluated in REAL*4 as the Fortran does; near a nodal plane (e.g. a station
    # at take-off 90 degrees) cos(90) is -4e-8 in single precision but 6e-17 in double, which
    # decides which side of the plane the station falls on
    rat_xyz32: np.ndarray = field(default_factory=lambda: np.zeros((0, 9), dtype=np.float32))
    rat_xyzden32: np.ndarray = field(default_factory=lambda: np.zeros((0, 6), dtype=np.float32))
    vpvs: float = 0.0

    @property
    def npol(self) -> int:
        return len(self.pol_sta)

    @property
    def nrat(self) -> int:
        return len(self.rat_sta)

    def count_pol(self, kind: int) -> int:
        return int(np.sum(self.pol_kind == kind))


def _float(s: str) -> float:
    s = s.strip()
    return float(s) if s else 0.0


def _field(line: str, lo: int, hi: int, decimals: int, what: str, lineno: int) -> float:
    """Read a fixed-column ``Fw.d`` field.

    Deliberate divergence from Fortran: a field written without a decimal point is read
    as written (``180`` is 180), where the original applies an implied decimal (``1.80``
    for F8.2), and blanks inside a number are an error where the original silently drops
    them.  The first case is reported with a warning.
    """
    raw = line[lo:hi]
    text = raw.strip()
    if not text:
        return 0.0
    try:
        value = float(text)
    except ValueError:
        raise InputError(f"line {lineno}: cannot read {what} from columns {lo + 1}-{hi}: {raw!r}") from None
    if "." not in text and "e" not in text.lower() and "d" not in text.lower() and value != 0.0:
        warnings.warn(
            f"line {lineno}: {what} {text!r} has no decimal point; read as {value:g} "
            f"(the Fortran program would read {value / 10 ** decimals:g})",
            stacklevel=3,
        )
    return value


_F4 = np.float32
_RD32 = _F4(45.0) / np.arctan(_F4(1.0))


def _vectors(azimuth: float, takeoff: float, single: bool = False):
    """Station direction vectors (focinp.f lines 335-352), optionally in REAL*4."""
    if single:
        trend = _F4(azimuth) / _RD32
        toa = _F4(takeoff) / _RD32
    else:
        trend = np.radians(azimuth)
        toa = np.radians(takeoff)
    ct, st = np.cos(trend), np.sin(trend)
    sp, cp = np.sin(toa), np.cos(toa)
    xyz = [ct * sp, st * sp, cp, -ct * cp, -st * cp, sp, st, -ct, 0.0]
    return xyz


def has_emergent_picks(lines) -> bool:
    """True if any line uses an emergent key (+ - l r f b) in column 21."""
    for k, line in enumerate(lines):
        if k == 0:
            continue
        # same rule as parse_inp: the key is column 21 of every line, whatever its layout
        line = line.rstrip("\n")[:80].rstrip()
        if len(line) > 20 and line[20] in EMERGENT_KEYS:
            return True
    return False


def parse_inp(source, emergent: bool | None = None, filename: str = "") -> FocmecData:
    """Parse FOCMEC input from a path, file object or iterable of lines.

    ``emergent`` decides whether emergent picks (``+ - l r``) are used; ``None``
    means "include them" (the default answer of the original prompt).
    """
    if isinstance(source, (str, os.PathLike)):
        filename = filename or os.fspath(source)
        with open(source, "r", errors="replace") as fh:
            lines = fh.read().splitlines()
    elif isinstance(source, io.IOBase) or hasattr(source, "read"):
        lines = source.read().splitlines()
    else:
        lines = [ln.rstrip("\n") for ln in source]
    if not lines:
        raise ValueError("empty FOCMEC input")

    d = FocmecData(filename=filename, comment=lines[0][:80].rstrip())
    d.include_emergent = emergent
    pguess: bool | None = None
    pol_sta, pol_sign, pol_kind, pol_xyz = [], [], [], []
    rat_sta, rat_sense, rat_numpol, rat_log, rat_xyz, rat_den = [], [], [], [], [], []
    pol_xyz32, rat_xyz32, rat_den32 = [], [], []
    maxdata = 1000  # MAX2 in FOCMEC.INC
    j = 0
    ratlog, snumpol = 0.0, " "
    for lineno, raw in enumerate(lines[1:], start=2):
        line = raw[:80].rstrip()  # the Fortran reads each record into CHARACTER*80
        ncom = len(line)
        line_p = line.ljust(60)
        sta = line_p[0:4]
        sense = line_p[20]
        if ncom < 22 or line[23:29].strip() == "":
            azin = _field(line_p, 4, 12, 2, "azimuth", lineno)
            toang1 = _field(line_p, 12, 20, 2, "take-off angle", lineno)
            info = line[39:ncom][:40].rstrip() if ncom > 21 else ""
            toang2 = 0.0
            if sense in "VSH":
                # Fortran silently reuses the ratio, numerator polarity and denominator
                # take-off angle of the previous line here (or garbage on the first one)
                raise InputError(
                    f"line {lineno}: ratio line for {sta.strip() or '?'} has no ratio data "
                    "(columns 24-29 are blank)"
                )
        else:
            azin = _field(line_p, 4, 12, 2, "azimuth", lineno)
            toang1 = _field(line_p, 12, 20, 2, "take-off angle", lineno)
            ratlog = _field(line_p, 21, 29, 4, "log ratio", lineno)
            snumpol = line_p[30]
            toang2 = _field(line_p, 32, 38, 2, "denominator take-off angle", lineno)
            d.vpvs = _field(line_p, 39, 45, 4, "Vp/Vs", lineno)
            info = line[45:ncom][:40].rstrip() if ncom > 45 else ""

        if sense in "VSH":
            j += 1
            if j > maxdata:
                break
            rat_sta.append(sta)
            rat_sense.append(sense)
            rat_numpol.append(snumpol)
            rat_log.append(ratlog)
            echo = rec(3, sta, 10, fnum(azin, 5, 1), 19, fnum(toang1, 5, 1), 27, sense,
                       32, fnum(ratlog, 8, 4), 41, "S", sense, 47, snumpol,
                       53, fnum(toang2, 6, 2), 60, info)
            d.echo.append(echo)
            rat_den.append(_vectors(azin, toang2)[:6])
            rat_xyz.append(_vectors(azin, toang1))
            rat_den32.append(_vectors(azin, toang2, True)[:6])
            rat_xyz32.append(_vectors(azin, toang1, True))
            continue

        d.echo.append(rec(3, sta, 10, fnum(azin, 5, 1), 19, fnum(toang1, 5, 1), 27, sense, 60, info))
        if sense == "U":
            sense = _P_UP
        if sense == "R":
            sense = ">"
        if sense == "L":
            sense = "<"
        if sense in EMERGENT_KEYS:
            d.has_emergent = True
            if pguess is None:
                pguess = True if emergent is None else bool(emergent)
                d.emergent_msg_index = len(d.echo) - 1
            if pguess:
                sense = _EMERGENT_MAP[sense]
        if sense not in "CDFB<>":
            continue
        j += 1
        if j > maxdata:
            break
        kind = 2 if sense in "<>" else 1 if sense in "FB" else 0
        pol_sta.append(sta)
        pol_kind.append(kind)
        pol_sign.append(1.0 if sense in "CF<" else -1.0)
        pol_xyz.append(_vectors(azin, toang1))
        pol_xyz32.append(_vectors(azin, toang1, True))

    d.include_emergent = pguess if pguess is not None else emergent
    d.pol_sta = pol_sta
    d.pol_sign = np.array(pol_sign, dtype=float)
    d.pol_kind = np.array(pol_kind, dtype=int)
    d.pol_xyz = np.array(pol_xyz, dtype=float).reshape(-1, 9)
    d.pol_xyz32 = np.array(pol_xyz32, dtype=_F4).reshape(-1, 9)
    d.rat_sta = rat_sta
    d.rat_sense = rat_sense
    d.rat_numpol = rat_numpol
    d.rat_log = np.array(rat_log, dtype=float)
    d.rat_xyz = np.array(rat_xyz, dtype=float).reshape(-1, 9)
    d.rat_xyzden = np.array(rat_den, dtype=float).reshape(-1, 6)
    d.rat_xyz32 = np.array(rat_xyz32, dtype=_F4).reshape(-1, 9)
    d.rat_xyzden32 = np.array(rat_den32, dtype=_F4).reshape(-1, 6)
    return d


def read_inp(path, emergent: bool | None = None) -> FocmecData:
    """Read a FOCMEC ``.inp`` file."""
    return parse_inp(path, emergent=emergent)


def _fmt_polarity_line(sta, az, toa, key, info=""):
    return f"{sta[:4]:<4}{az:8.2f}{toa:8.2f}{key}" + (f"{'':18}{info}" if info else "")


def format_polarity(sta, azimuth, takeoff, key, info="") -> str:
    """One polarity line in ``.inp`` format."""
    return _fmt_polarity_line(sta, azimuth, takeoff, key, info)


def format_ratio(sta, azimuth, takeoff, sense, lograt, numpol, den_takeoff, vpvs, info="") -> str:
    """One amplitude-ratio line in ``.inp`` format."""
    s = (
        f"{sta[:4]:<4}{azimuth:8.2f}{takeoff:8.2f}{sense}{lograt:8.4f} {numpol} "
        f"{den_takeoff:6.2f} {vpvs:6.4f}"
    )
    return s + (f" {info}" if info else "")


def write_inp(path, comment: str, lines) -> None:
    """Write a FOCMEC input file from a comment and preformatted lines."""
    with open(path, "w") as fh:
        fh.write(comment.rstrip() + "\n")
        for ln in lines:
            fh.write(ln.rstrip() + "\n")


def convert_legacy(lines, vpvs: float) -> list[str]:
    """Convert a pre-2017 file by inserting ``vpvs`` at columns 40-45 of every
    ratio line (port of ``convert_input.f``).  The first line is the comment.
    Any trailing comment of an old ratio line (from column 39) follows the new
    field."""
    out = [lines[0].rstrip("\n")]
    for ln in lines[1:]:
        ln = ln.rstrip("\n").rstrip()
        if len(ln) >= 22 and ln[23:29].strip() != "":
            ln = ln[:38].ljust(38) + f" {vpvs:6.4f}" + ln[38:]
        out.append(ln)
    return out


@dataclass
class Pick:
    """One data line of a FOCMEC input file, as written (nothing filtered)."""

    sta: str
    azimuth: float
    takeoff: float
    code: str
    log_ratio: float | None = None
    numerator_polarity: str = ""
    takeoff_den: float | None = None
    info: str = ""

    @property
    def is_ratio(self) -> bool:
        return self.code in "VHS"


def read_picks(source) -> tuple[str, list[Pick]]:
    """Every data line of a ``.inp`` file (including emergent picks) for plotting.

    Returns ``(comment, picks)``.
    """
    if isinstance(source, (str, os.PathLike)):
        with open(source, errors="replace") as fh:
            lines = fh.read().splitlines()
    else:
        lines = [ln.rstrip("\n") for ln in source]
    picks = []
    for raw in lines[1:]:
        line = raw[:80].rstrip()
        if not line.strip():
            continue
        ncom = len(line)
        p = line.ljust(60)
        sta, az, toa, code = p[0:4].strip(), _float(p[4:12]), _float(p[12:20]), p[20]
        if ncom >= 22 and line[23:29].strip():
            picks.append(Pick(sta, az, toa, code, _float(p[21:29]), p[30], _float(p[32:38]),
                              line[45:ncom].strip() if ncom > 45 else ""))
        else:
            picks.append(Pick(sta, az, toa, code, info=line[39:ncom].strip() if ncom > 21 else ""))
    return lines[0].rstrip(), picks
