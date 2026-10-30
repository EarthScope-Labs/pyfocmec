"""``focmec`` command line: flags/TOML for scripting, ``--interactive`` for the original prompts."""

from __future__ import annotations

import argparse
import sys
import time

from .. import __version__
from ..inp import parse_inp
from ..params import FocmecParams
from ..prompts import Prompter, PromptEOF, interactive_params
from ..report import out_lines
from ..search import search
from .errors import clean_errors


def _triplet(vals):
    lo, step, hi = vals
    return (float(lo), float(step), None if str(hi).lower() in ("default", "none") else float(hi))


def build_parser():
    p = argparse.ArgumentParser(
        prog="focmec",
        description="Find acceptable double-couple focal mechanisms from polarities and amplitude ratios.",
    )
    p.add_argument("input", nargs="?", help="FOCMEC .inp file (not used with --interactive)")
    p.add_argument("-i", "--interactive", action="store_true",
                   help="replay the original FOCMEC prompts, reading answers from standard input")
    p.add_argument("--params", help="TOML file with search parameters (FocmecParams field names)")
    p.add_argument("-c", "--comment", help="comment line written to the output files")
    p.add_argument("-o", "--out", help="summary output file [focmec.out]")
    p.add_argument("--lst", default="focmec.lst", help="detailed listing file [focmec.lst]")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--rw", dest="relweight", action="store_true", default=None,
                   help="relative weighting of polarity errors (default)")
    g.add_argument("--uw", dest="relweight", action="store_false", help="unweighted polarity errors")
    p.add_argument("--p-errors", type=float, help="allowed P polarity errors")
    p.add_argument("--sv-errors", type=float, help="allowed SV polarity errors")
    p.add_argument("--sh-errors", type=float, help="allowed SH polarity errors")
    p.add_argument("--total-errors", type=float, help="allowed total polarity errors (P with SV/SH data)")
    p.add_argument("--ratio-error", type=float, help="max |log10 observed - calculated| ratio error [0.6]")
    p.add_argument("--n-ratio-errors", type=int, help="number of ratio errors allowed [0]")
    p.add_argument("--cutp", type=float, help="lower limit of P radiation factor for ratios [0.1]")
    p.add_argument("--cuts", type=float, help="lower limit of S radiation factor for ratios [0.1]")
    p.add_argument("--maxsol", type=int, help="stop after this many acceptable solutions [100]")
    for name in ("trend", "plunge", "angle"):
        p.add_argument(f"--{name}", nargs=3, metavar=("MIN", "INC", "MAX"),
                       help=f"search range for B {name}" if name != "angle" else "search range for Angle (degrees; MAX may be 'default')")
    e = p.add_mutually_exclusive_group()
    e.add_argument("--emergent", dest="emergent", action="store_true", default=None,
                   help="include emergent polarity picks (default)")
    e.add_argument("--no-emergent", dest="emergent", action="store_false", help="ignore emergent picks")
    p.add_argument("--float32", action="store_true", default=None,
                   help="compute in single precision like the Fortran original (default with --interactive)")
    p.add_argument("--float64", dest="float32", action="store_false", help="compute in double precision (default without --interactive)")
    p.add_argument("-q", "--quiet", action="store_true", help="do not print solutions to the terminal")
    p.add_argument("--version", action="version", version=f"pyfocmec {__version__}")
    return p


def _load_toml(path):
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def params_from_args(args) -> FocmecParams:
    prm = FocmecParams()
    if args.params:
        for k, v in _load_toml(args.params).items():
            if not hasattr(prm, k):
                raise SystemExit(f"focmec: unknown parameter {k!r} in {args.params}")
            if k in ("trend", "plunge", "angle"):
                v = _triplet(v)
            setattr(prm, k, v)
    mapping = dict(
        relweight=args.relweight, p_errors=args.p_errors, sv_errors=args.sv_errors,
        sh_errors=args.sh_errors, total_errors=args.total_errors, ratio_error=args.ratio_error,
        n_ratio_errors=args.n_ratio_errors, cutp=args.cutp, cuts=args.cuts, maxsol=args.maxsol,
        comment=args.comment, out_name=args.out, emergent=args.emergent,
    )
    for k, v in mapping.items():
        if v is not None:
            setattr(prm, k, v)
    for name in ("trend", "plunge", "angle"):
        v = getattr(args, name)
        if v:
            setattr(prm, name, _triplet(v))
    return prm


@clean_errors("focmec")
def main(argv=None, stdin=None, stdout=None):
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    args = build_parser().parse_args(argv)
    import numpy as np

    use32 = args.interactive if args.float32 is None else args.float32
    dtype = np.float32 if use32 else np.float64
    if args.interactive:
        pr = Prompter(stdin, out=stdout)
        stamp_t = time.localtime()
        try:
            data, prm = interactive_params(pr)
        except PromptEOF as exc:
            print(f"focmec: {exc}", file=sys.stderr)
            return 1
        if args.out:
            prm.out_name = args.out
    else:
        if not args.input:
            build_parser().error("an input file is required (or use --interactive)")
        prm = params_from_args(args)
        data = parse_inp(args.input, emergent=prm.emergent)
        stamp_t = time.localtime()
    result = search(data, prm, dtype=dtype)
    result.timestamp = (
        time.strftime("%a %b ", stamp_t) + f"{stamp_t.tm_mday:2d}" + time.strftime(" %H:%M:%S %Y", stamp_t)
    )
    result.write_lst(args.lst)
    result.write_out(prm.out_name)
    if not args.quiet:
        _print_summary(result, stdout)
    return 0


def _print_summary(result, stdout):
    lines = out_lines(result)
    ndata = len(lines) - result.nsol
    for ln in lines[ndata - 1:]:
        stdout.write(ln + "\n")
    if result.truncated:
        stdout.write(f" Reached chosen maximum of{result.resolved.maxsol:5d} solutions\n")
    if result.nsol:
        stdout.write(f"\n There are{result.nsol:4d} acceptable solutions\n")
    else:
        stdout.write("\n There are no acceptable solutions\n")


if __name__ == "__main__":
    sys.exit(main())
