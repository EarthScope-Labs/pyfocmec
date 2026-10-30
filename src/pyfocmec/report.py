"""Writers for FOCMEC's ``focmec.lst`` (full detail) and ``.out`` (summary) files.

The text layout reproduces the Fortran formats of ``focinp.f``, ``oksol.f`` and
``focreps.f`` so that existing parsers and plotting workflows keep working.
"""

from __future__ import annotations


from .fortfmt import fnum, inum, rec
from .radiation import FLAG_ND, FLAG_TEXT

SEP76 = " " + "+" * 76
SEP80 = "+" * 80


def _grid_lines(res):
    bt, bp, aa = res.bt, res.bp, res.aa
    return [
        " The minimum, increment and maximum B axis trend: " + "".join(fnum(x, 8, 2) for x in bt),
        " The limits for the B axis plunge: " + "".join(fnum(x, 8, 2) for x in bp),
        " The limits for Angle: " + "".join(fnum(x, 8, 2) for x in aa),
    ]


def _criteria_lines(result):
    """Lines written to both files after the input has been read."""
    d, res = result.data, result.resolved
    lines = []
    if d.npol <= 0:
        lines.append(" There are no polarity data")
    elif res.total_mode:
        if res.relweight:
            lines.append(
                f"{inum(res.nppol, 4)} P Pol.{inum(res.nsvpol, 3)} SV Pol.{inum(res.nshpol, 3)} SH Pol."
                f"{fnum(res.err, 5, 1)} allowed (weighted) errors"
            )
        else:
            lines.append(
                f"{inum(res.nppol, 4)} P Pol.{inum(res.nsvpol, 3)} SV Pol.{inum(res.nshpol, 3)} SH Pol."
                f"{inum(int(res.err), 4)} allowed  errors"
            )
    elif res.thresh < 1.0:
        lines.append(
            f" Polarities/Errors:  P {inum(res.nppol, 3, True)}/{fnum(res.errp, 4, 1)}"
            f"  SV {inum(res.nsvpol, 3, True)}/{fnum(res.errsv, 4, 1)}"
            f"  SH {inum(res.nshpol, 3, True)}/{fnum(res.errsh, 4, 1)}"
        )
    else:
        lines.append(
            f" Polarities/Errors:  P {inum(res.nppol, 3, True)}/{inum(int(res.errp), 2, True)}"
            f"  SV {inum(res.nsvpol, 3, True)}/{inum(int(res.errsv), 2, True)}"
            f"  SH {inum(res.nshpol, 3, True)}/{inum(int(res.errsh), 2, True)}"
        )
    if d.nrat <= 0:
        lines.append(" There are no amplitude ratio data")
    else:
        lines.append(
            f"{inum(d.nrat, 3)} ratios, maximum of{inum(res.nerrr, 3)} with |o-c| diff > "
            f"{fnum(res.errrat, 7, 4)}    Focus VP/VS: {fnum(d.vpvs, 6, 4)}"
        )
        lines.append(
            f"For ratios,  {fnum(res.cutp, 5, 3)} = P radiation cutoff  "
            f"{fnum(res.cuts, 5, 3)} = S radiation cutoff"
        )
    return lines


_RATIO_NOTES = [
    " FLAG is NUM, DEN, N&D if n, d, both below curoff",
    " If FLAG N&D, total ratios used decreased by 1",
    " R Ac/Tot: # acceptable / total ratios minus M&D",
    " RMS Err: RMS of acceptable obs ratios",
    " AbsMaxDiff: max abs difference for ok solutions",
]

_OUT_HEADER_RATIO = rec(
    5, "Dip", 11, "Strike", 20, "Rake", 28, "Pol:", 33, "P", 39, "SV", 45, "SH",
    49, "AccR/TotR", 60, "RMS RErr", 70, "AbsMaxDiff",
)
_OUT_HEADER_POL = rec(5, "Dip", 11, "Strike", 20, "Rake", 28, "Pol:", 33, "P", 39, "SV", 45, "SH")


def _preamble(result, stamp):
    d = result.data
    lines = [f"  {stamp} for program Focmec", " " + result.params.comment, f" Input from a file {d.filename}",
             " " + d.comment]
    return lines


def _emergent_msg(d):
    if not d.has_emergent:
        return None
    return " Including emergent polarity picks" if d.include_emergent else " Not including emergent polarity picks"


def _row_out(sol, d):
    parts = [fnum(sol.dip, 8, 2) + fnum(sol.strike, 8, 2) + fnum(sol.rake, 8, 2)]
    line = rec(1, parts[0], 30, fnum(sol.bad_p, 6, 2) + fnum(sol.bad_sv, 6, 2) + fnum(sol.bad_sh, 6, 2))
    if d.nrat > 0:
        line = rec(
            1, line, 52,
            f"{inum(sol.ratio_ok, 2, True)}/{inum(sol.ratio_total, 2, True)}"
            f"{fnum(sol.ratio_rms, 10, 4)}{fnum(sol.ratio_maxdiff, 10, 4)}",
        )
    return line


def _stations(items, sep=" "):
    return sep.join(s.ljust(4)[:4] for s in items)


def _weights(items):
    return " ".join(fnum(w, 4, 2) for w in items)


def _pol_block(sol, kind):
    """Polarity-error listing for one phase, including the original's layout quirks."""
    errs = sol.bad_stations[kind]
    n = len(errs)
    if n == 0:
        return []
    stas = [s for s, _ in errs]
    ws = [w for _, w in errs]
    label_s = {"P": " P Polarity error at", "SV": " SV Polarity error at", "SH": " SH Polarity error at"}[kind]
    label_w = {"P": " P Polarity weights:", "SV": " SV Polarity weights:", "SH": " SH Polarity weights:"}[kind]
    tab1 = 23 if kind == "P" else 24
    total = {
        "P": ("  Total P polarity weight is", "   Total number: ", sol.bad_p),
        "SV": ("  Total SV polarity weight is", "  Total number: ", sol.bad_sv),
        "SH": ("  Total SH polarity weight is", "  Total number: ", sol.bad_sh),
    }[kind]
    out = [""]
    out.append(rec(label_s, tab1, _stations(stas[:11])))
    out.append(rec(label_w, tab1, _weights(ws[:11])))
    if n > 11:
        # second group: stations 12-22
        out.append(rec(23, _stations(stas[11:22])))
        if kind == "SV":
            out.append(rec(label_w, tab1, _weights(ws[11:22])))
        else:
            out.append(rec(23, _weights(ws[11:22])))
        if n > 22:
            for i in range(22, n, 11):
                out.append(rec(23, _stations(stas[i:i + 11])))
            for i in range(22, n, 11):
                out.append(rec(23, _weights(ws[i:i + 11])))
    out.append(total[0] + fnum(total[2], 7, 3) + total[1] + inum(n, 3))
    return out


_FLAG_POL = {"V": ("F", "B"), "S": ("F", "B"), "H": ("L", "R")}


def _lst_solution(sol, d, res):
    lines = []
    lines.append("     Dip,Strike,Rake " + "".join(fnum(x, 9, 2) for x in (sol.dip, sol.strike, sol.rake)))
    lines.append(
        "     Dip,Strike,Rake " + "".join(fnum(x, 9, 2) for x in (sol.aux_dip, sol.aux_strike, sol.aux_rake))
        + "   Auxiliary Plane"
    )
    lines.append(
        "     Lower Hem. Trend, Plunge of A,N "
        + "".join(fnum(x, 9, 2) for x in (sol.a_trend, sol.a_plunge, sol.n_trend, sol.n_plunge))
    )
    lines.append(
        "     Lower Hem. Trend, Plunge of P,T "
        + "".join(fnum(x, 9, 2) for x in (sol.p_trend, sol.p_plunge, sol.t_trend, sol.t_plunge))
    )
    lines.append(
        "     B trend, B plunge, Angle: "
        + "".join(fnum(x, 7, 2) for x in (sol.b_trend, sol.b_plunge, sol.angle))
    )
    for kind in ("P", "SV", "SH"):
        lines.extend(_pol_block(sol, kind))
    if d.nrat <= 0:
        lines.append(SEP80)
        lines.append("")
        return lines
    lines.append("")
    lines.append(rec(11, "Log10(Ratio)", 53, "Ratio", 63, "S Polarity"))
    lines.append(rec(6, "Observed", 16, "Calculated", 30, "Difference", 42, "Station", 53, " Type",
                     63, "Obs.", 69, "Calc.", 75, "Flag"))
    pol = " "
    for k, r in enumerate(sol.ratios):
        sense = r["sense"]
        if r["spol"] > 0.0:
            pol = _FLAG_POL[sense][0]
        elif r["spol"] < 0.0:
            pol = _FLAG_POL[sense][1]
        flag = FLAG_TEXT[r["flag"]]
        if r["flag"] != FLAG_ND:
            mark = "*" if r["weight"] == 0 else " "
            lines.append(rec(5, fnum(r["observed"], 8, 4), 17, fnum(r["calculated"], 8, 4),
                             30, fnum(r["difference"], 8, 4), 43, mark, r["station"], 55, "S", sense,
                             64, r["numpol"], 71, pol, 76, flag))
        else:
            lines.append(rec(5, fnum(r["observed"], 8, 4), 43, "#", r["station"], 55, "S", sense,
                             64, r["numpol"], 71, pol, 76, flag))
    lines.append("")
    lines.append(f"Total number of ratios used is {inum(sol.ratio_total, 3)}")
    lines.append(f"RMS error for the{inum(sol.ratio_ok, 3)} acceptable solutions is{fnum(sol.ratio_rms, 7, 4)}")
    lines.append("  Highest absolute velue of diff for those solutions is " + fnum(sol.ratio_maxdiff, 7, 4))
    lines.append(SEP80)
    lines.append("")
    return lines


def lst_lines(result) -> list[str]:
    d, res = result.data, result.resolved
    lines = _preamble(result, result.timestamp)
    lines += ["", rec(2, "Statn", 9, "Azimuth", 20, "TOA", 26, "Key", 31, "Log10 Ratio", 44, "NumPol",
                       52, "DenTOA", 60, "Comment")]
    msg = _emergent_msg(d)
    for i, e in enumerate(d.echo):
        lines.append(e)
        if msg and d.emergent_msg_index == i:
            lines.append(msg)
    lines += _criteria_lines(result)
    if d.nrat > 0:
        lines += _RATIO_NOTES[:2]
    lines += _grid_lines(res)
    lines += [SEP76, ""]
    for sol in result.solutions:
        lines += _lst_solution(sol, d, res)
    if result.truncated:
        lines.append(f" Reached chosen maximum of{inum(res.maxsol, 5)} solutions")
    if result.nsol > 0:
        lines += ["", f" There are{inum(result.nsol, 4)} acceptable solutions"]
    else:
        lines += ["", " There are no acceptable solutions"]
    return lines


def out_lines(result) -> list[str]:
    d, res = result.data, result.resolved
    lines = _preamble(result, result.timestamp)
    msg = _emergent_msg(d)
    if msg:
        lines.append(msg)
    lines += _criteria_lines(result)
    if d.nrat > 0:
        lines += _RATIO_NOTES
    lines += _grid_lines(res)
    lines.append(_OUT_HEADER_RATIO if d.nrat > 0 else _OUT_HEADER_POL)
    for sol in result.solutions:
        lines.append(_row_out(sol, d))
    return lines


def _write(lines, dest):
    text = "\n".join(lines) + "\n"
    if hasattr(dest, "write"):
        dest.write(text)
        return
    with open(dest, "w") as fh:
        fh.write(text)


def write_lst(result, dest="focmec.lst"):
    """Write the detailed listing (``focmec.lst``)."""
    _write(lst_lines(result), dest)


def write_out(result, dest="focmec.out"):
    """Write the summary file used by ``focplt`` (``focmec.out``)."""
    _write(out_lines(result), dest)
