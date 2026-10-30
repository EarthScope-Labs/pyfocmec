"""Replay of the original FOCMEC interactive dialogue (``focinp.f`` + prompt helpers).

The Fortran helpers ``cvalue``/``rvalue``/``ivalue``/``truth`` read one line per
prompt from standard input:

* lines starting with ``/*`` are comments and skipped,
* a TAB ends the answer (so scripts can append notes after a tab),
* an empty answer selects the default shown in brackets.

This lets the ``r*`` driver scripts shipped with FOCMEC run unchanged.
"""

from __future__ import annotations

import re
import sys

from .fortfmt import rec  # noqa: F401  (re-exported for convenience)
from .inp import has_emergent_picks, parse_inp
from .params import FocmecParams

_FLOAT = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eEdD][+-]?\d+|[+-]\d+)?$")


class PromptEOF(EOFError):
    """Standard input ended while FOCMEC was still waiting for an answer."""


def printx(msg: str, out) -> None:
    """Prompt text as drawn by ``printx.f`` (dots out to column 49, then ': ')."""
    msg = msg.rstrip()
    if msg.endswith("$"):
        msg = msg[:-1]
    kk = min(len(msg), 59 if len(msg) > 58 else len(msg))
    buf = msg[:kk]
    if kk < 49:
        buf = buf + "." * (49 - kk)
        kk = 49
    out.write(" " + buf[: kk - 1] + ": ")
    out.flush()


class Prompter:
    def __init__(self, lines, out=None):
        self._it = iter(lines)
        self.out = out if out is not None else sys.stdout

    def _read(self, width: int | None = None) -> str:
        """``cstring``: next non-comment line, cut at TAB, trailing blanks removed."""
        while True:
            try:
                line = next(self._it)
            except StopIteration:
                raise PromptEOF("end of input while waiting for an answer") from None
            line = line.rstrip("\n")
            if width is not None:
                line = line[:width]
            line = line.rstrip(" \x00")
            if not line.startswith("/*"):
                break
        tab = line.find("\t")
        if tab >= 0:
            line = line[:tab].rstrip(" ")  # the Fortran string ends at the TAB; blanks before it are padding
        return line

    def cvalue(self, msg: str, default: str = "") -> str:
        printx(msg, self.out)
        s = self._read(80).lstrip(" ")
        return s if s else default

    def truth(self, msg: str) -> bool:
        printx(msg, self.out)
        s = self._read(80)
        if s and s[0] == " ":
            s = ""
        if not s:
            return msg.rstrip("$").rstrip()[-3:] in ("[Y]", "[y]")
        return s[0] in "Yy"

    def _number(self, msg: str, default):
        while True:
            printx(msg, self.out)
            s = self._read(30)
            if s:
                i = s.find(" ")
                if i >= 0:
                    s = s[:i]
            if not s:
                return default
            if len(s) == 1:
                s = "0" + s
            tok = s.split(",")[0]
            if _FLOAT.match(tok):
                # Fortran list-directed input also accepts "5.6+01" and a D exponent
                tok = re.sub(r"(?<=[\d.])([+-]\d+)$", r"e\1", tok).replace("d", "e").replace("D", "e")
                return float(tok)
            # unreadable: the Fortran ERR= branch prompts again

    def rvalue(self, msg: str, default: float) -> float:
        return float(self._number(msg, default))

    def ivalue(self, msg: str, default: int) -> int:
        from .params import nint

        v = self._number(msg, default)
        return nint(v) if not isinstance(v, int) else v

    def numbers(self, msg: str, count: int) -> list[float]:
        """List-directed ``READ(*,*)`` of ``count`` numbers (may span several lines)."""
        printx(msg, self.out)
        vals: list[float] = []
        while len(vals) < count:
            try:
                line = next(self._it)
            except StopIteration:
                raise PromptEOF("end of input while waiting for numbers") from None
            for tok in re.split(r"[\s,]+", line.strip()):
                if not tok:
                    continue
                t = re.sub(r"(?<=[\d.])([+-]\d+)$", r"e\1", tok).replace("d", "e").replace("D", "e")
                try:
                    vals.append(float(t))
                except ValueError:
                    break  # the rest of the record is not numeric
                if len(vals) == count:
                    break
        return vals

    def getstring(self, msg: str) -> str:
        printx(msg, self.out)
        return self._read(80).lstrip(" ")

    def value(self, msg: str) -> float:
        return float(self._number(msg, 0.0))


def interactive_params(prompter: Prompter, base: FocmecParams | None = None):
    """Run the prompt sequence of ``focinp.f``.

    Returns ``(data, params)`` ready for :func:`pyfocmec.search`.
    """
    import os

    pr = prompter
    params = base or FocmecParams()
    params.out_name = pr.cvalue("Output file name (for plotting) [focmec.out]", "focmec.out")
    params.comment = pr.cvalue("Comment - up to 80 characters", "")
    while True:
        fname = pr.cvalue("Input filespec", "")
        if not fname or not os.path.isfile(fname):
            continue
        with open(fname, errors="replace") as fh:
            lines = fh.read().splitlines()
        if not lines:
            continue
        pr.out.write(" " + lines[0].rstrip() + "\n")
        if pr.truth("Correct file?...[Y]"):
            break
    emergent = None
    if has_emergent_picks(lines):
        emergent = pr.truth("Include emergent polarity picks?..[Y]")
    data = parse_inp(lines, emergent=emergent, filename=fname)
    params.emergent = emergent
    nppol, nsv, nsh = (data.count_pol(k) for k in (0, 1, 2))
    pr.out.write(
        f" Input:{nppol:4d} P {nsv:4d} SV and {nsh:4d} SH polarities and,{data.nrat:4d} ratios\n"
    )
    if data.npol > 0:
        pr.out.write(" Can have relative weighting for polarity errors\n")
        pr.out.write("  for which weight = theor. rad. factor (0 to 1)\n")
        params.relweight = pr.truth("Relative weighting?..[Y]")
        total_done = False
        if nppol > 0 and (nsh > 0 or nsv > 0):
            pr.out.write(" Options:  (1) Total polarity errors\n")
            pr.out.write("           (2) Separate P, SV, and SH (default)\n")
            if pr.truth("Total polarity error option?..[N]"):
                msg = (
                    "Total number of errors (floating point)"
                    if params.relweight
                    else "Total number of errors (integer)"
                )
                params.total_errors = pr.value(msg)
                total_done = True
        if not total_done:
            params.total_errors = None
            if params.relweight:
                if nppol > 0:
                    params.p_errors = pr.rvalue("Allowed P polarity errors..[0.0]", 0.0)
                if nsv > 0:
                    params.sv_errors = pr.rvalue("Allowed SV polarity errors..[0.0]", 0.0)
                if nsh > 0:
                    params.sh_errors = pr.rvalue("Allowed SH polarity errors..[0.0]", 0.0)
            else:
                if nppol > 0:
                    params.p_errors = pr.ivalue("Allowed P polarity errors..[0]", 0)
                if nsv > 0:
                    params.sv_errors = pr.ivalue("Allowed SV polarity errors..[0]", 0)
                if nsh > 0:
                    params.sh_errors = pr.ivalue("Allowed SH polarity errors..[0]", 0)
    if data.nrat > 0:
        params.ratio_error = pr.rvalue("Max permitted log10 |o-c| error..[0.6]", 0.6)
        params.n_ratio_errors = pr.ivalue("Number permitted |o-c| errors..[0]", 0)
        pr.out.write(" Next two entries are for near-nodal amplitudes\n")
        pr.out.write(" CUTP is the lower bound for P radiation factor\n")
        pr.out.write(" CUTS is the lower bound for S radiation factor\n")
        pr.out.write(" Ratio is indeterminate if both calculated\n")
        pr.out.write("     values less than the chosen CUT values\n")
        params.cutp = pr.rvalue("CUTP: lower-limit P cutoff...[0.1]", 0.1)
        params.cuts = pr.rvalue("CUTS: lower-limit S cutoff...[0.1]", 0.1)
    params.maxsol = pr.ivalue("Exit after this many acceptable sols...[100]", 100)
    btmin = pr.rvalue("Minimum search value B trend..[0]", 0.0)
    btdel = abs(pr.rvalue("Increment for B trend..[5 degree]", 5.0))
    btmax = pr.rvalue("Maximum B trend..[360-btdel]", 360 - btdel)
    bpmin = pr.rvalue("Minimum search value B plunge..[0]", 0.0)
    bpdel = abs(pr.rvalue("Increment for B plunge..[5 degree]", 5.0))
    bpmax = pr.rvalue("Maximum B plunge..[90 degrees]", 90.0)
    pr.out.write(' "Angle" in vertical plane of B trend)\n')
    aamin = pr.rvalue("Minimum search value Angle..[0]", 0.0)
    aadel = abs(pr.rvalue("Increment for Angle..[5 degree]", 5.0))
    aamax = pr.rvalue("Maximum Angle..[180-aadel]", 180 - aadel)
    params.trend = (btmin, btdel, btmax)
    params.plunge = (bpmin, bpdel, bpmax)
    params.angle = (aamin, aadel, aamax)
    return data, params
