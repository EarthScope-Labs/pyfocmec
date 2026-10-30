"""``pyfocmec dsretc``: convert between dip/strike/rake, P/T axes and moment tensors."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .. import dsretc as ds
from ..fortfmt import fnum
from ..prompts import Prompter, PromptEOF
from .errors import clean_errors


def _stamp(prog):
    t = time.localtime()
    return "  " + time.strftime("%a %b ", t) + f"{t.tm_mday:2d}" + time.strftime(" %H:%M:%S %Y", t) + f" for program {prog}"


def analyse(dsr=None, pt=None, mt=None, dtype=np.float64):
    """Representations plus any moment-tensor notes (``ValueError`` args carry the notes)."""
    if dsr is not None:
        return ds.from_dsr(*dsr, dtype=dtype)
    if pt is not None:
        return ds.from_pt(*pt, dtype=dtype)
    rep, _ = ds.from_moment_tensor(*mt, dtype=dtype)
    return rep


def result_lines(rep, comment="", pt_input=None, ball=True, dtype=np.float32):
    """Listing block for one mechanism.  Like the original, the beachball is only
    drawn once a consistent FOCMEC Angle was found."""
    out = ["", "     " + comment]
    if pt_input is not None:
        out.append("     Input: Trend, Plunge of P,T " + "".join(fnum(x, 9, 2) for x in pt_input))
    out += rep.notes
    out += rep.lines()
    if rep.angle is not None:
        out.append("     Angle: " + fnum(rep.angle, 9, 2))
        if ball:
            out += ds.ascii_beachball(rep.moment, rep.p, rep.t, dtype=dtype)
    return out


def run_interactive(stdin, stdout, lst="dsretc.lst"):
    pr = Prompter(stdin, out=stdout)
    dtype = np.float32
    lines = [_stamp("dsretc")]
    ok = True
    try:
        comment = pr.getstring("Comment")
        while True:
            pt_input = None
            if pr.truth("Dip, Strike and Rake?..[Y]"):
                dsr = pr.numbers("Enter Dip, Strike and Rake (degrees)", 3)
                rep = ds.from_dsr(*dsr, dtype=dtype)
            elif pr.truth("P and T axes trend and plunge?..[Y]"):
                pt_input = pr.numbers("Enter trend and plunge for P and T (t,p,t,p)", 4)
                rep = ds.from_pt(*pt_input, dtype=dtype)
            else:
                mt = pr.numbers("Input MRR MTT MPP MRT MRP MTP (free format)", 6)
                try:
                    rep, _ = ds.from_moment_tensor(*mt, dtype=dtype)
                except ValueError as exc:
                    lines += ["", "     " + comment] + list(exc.args[1])
                    break
            block = result_lines(rep, comment, pt_input, dtype=dtype)
            lines += block
            for ln in block[1:]:
                stdout.write(ln + "\n")
            if rep.angle is None:
                break  # the original program stops when no consistent Angle exists
            if not pr.truth("Run some more?...[Y]"):
                break
            if not pr.truth("Same comment?..[Y]"):
                comment = pr.getstring("Comment")
                lines.append(_stamp("dsretc"))  # dsretc.f writes a new time stamp after each new comment
    except PromptEOF as exc:
        print(f"dsretc: {exc}", file=sys.stderr)
        ok = False
    with open(lst, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return ok


def build_parser():
    p = argparse.ArgumentParser(
        prog="pyfocmec dsretc",
        description="Convert a focal mechanism between dip/strike/rake, P/T axes and moment tensor, "
        "and compute the FOCMEC 'Angle' for a fixed-solution run.",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dsr", nargs=3, type=float, metavar=("DIP", "STRIKE", "RAKE"))
    g.add_argument("--pt", nargs=4, type=float, metavar=("P_TREND", "P_PLUNGE", "T_TREND", "T_PLUNGE"))
    g.add_argument("--mt", nargs=6, type=float, metavar=("MRR", "MTT", "MPP", "MRT", "MRP", "MTP"),
                   help="moment tensor (Dziewonski/Harvard CMT convention)")
    p.add_argument("-i", "--interactive", action="store_true", help="replay the original dsretc prompts from stdin")
    p.add_argument("-c", "--comment", default="")
    p.add_argument("--lst", default=None, help="also write the listing to this file (default dsretc.lst with --interactive)")
    p.add_argument("--no-ball", action="store_true", help="omit the ASCII beachball")
    p.add_argument("--float32", action="store_true", help="single precision like the Fortran original")
    return p


@clean_errors("pyfocmec dsretc")
def main(argv=None, stdin=None, stdout=None):
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    args = build_parser().parse_args(argv)
    if args.interactive:
        return 0 if run_interactive(stdin, stdout, args.lst or "dsretc.lst") else 1
    if not (args.dsr or args.pt or args.mt):
        build_parser().error("give --dsr, --pt or --mt (or --interactive)")
    dtype = np.float32 if args.float32 else np.float64
    try:
        rep = analyse(args.dsr, args.pt, args.mt, dtype)
    except ValueError as exc:
        for ln in exc.args[1]:
            stdout.write(ln + "\n")
        return 1
    lines = result_lines(rep, args.comment, args.pt, ball=not args.no_ball, dtype=dtype)
    for ln in lines[1:]:
        stdout.write(ln + "\n")
    if args.lst:
        with open(args.lst, "w") as fh:
            fh.write("\n".join([_stamp("dsretc")] + lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
