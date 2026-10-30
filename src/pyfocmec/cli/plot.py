"""Plotting commands: focplt and ratplt."""

from __future__ import annotations

import argparse
import sys

from .errors import clean_errors


@clean_errors("pyfocmec focplt")
def focplt_main(argv=None, stdout=None):
    stdout = stdout or sys.stdout
    p = argparse.ArgumentParser(prog="pyfocmec focplt", description="Plot polarities and focal mechanism solutions.")
    p.add_argument("--inp", help="FOCMEC input file (polarities)")
    p.add_argument("--out", help="focmec.out file with solutions")
    p.add_argument("--dsr", nargs=3, type=float, metavar=("DIP", "STRIKE", "RAKE"), help="highlight one mechanism")
    p.add_argument("--upper", action="store_true", help="upper hemisphere")
    p.add_argument("--phases", nargs="+", default=["P", "SV", "SH"], choices=["P", "SV", "SH"])
    p.add_argument("--emergent", action="store_true", help="also plot emergent picks")
    p.add_argument("--nodal-surfaces", action="store_true", help="SV and SH nodal surfaces of --dsr")
    p.add_argument("--beachball", action="store_true", help="shade the compressional quadrants of --dsr")
    p.add_argument("--labels", action="store_true", help="label stations")
    p.add_argument("--title")
    p.add_argument("-o", "--output", default="focplt.pdf", help="image file (.pdf, .png, .svg, .eps) [focplt.pdf]")
    args = p.parse_args(argv)
    if not (args.inp or args.out or args.dsr):
        p.error("give --inp, --out and/or --dsr")
    import matplotlib

    matplotlib.use("Agg")
    from ..plot import focplt

    ax = focplt(args.inp, args.out, args.dsr, upper=args.upper, phases=tuple(args.phases), emergent=args.emergent,
                nodal_surfaces=args.nodal_surfaces, beachball=args.beachball, title=args.title, labels=args.labels)
    ax.figure.savefig(args.output, bbox_inches="tight", dpi=200)
    stdout.write(f"wrote {args.output}\n")
    return 0


@clean_errors("pyfocmec ratplt")
def ratplt_main(argv=None, stdout=None):
    stdout = stdout or sys.stdout
    p = argparse.ArgumentParser(prog="pyfocmec ratplt", description="Plot amplitude-ratio misfit for a mechanism.")
    p.add_argument("inp", help="FOCMEC input file with ratios")
    p.add_argument("--dsr", nargs=3, type=float, required=True, metavar=("DIP", "STRIKE", "RAKE"))
    p.add_argument("--error", type=float, default=0.1, help="reference log10 |o-c| [0.1]")
    p.add_argument("--cutp", type=float, default=0.1)
    p.add_argument("--cuts", type=float, default=0.1)
    p.add_argument("--scale", type=float, default=0.20)
    p.add_argument("--upper", action="store_true")
    p.add_argument("-o", "--output", default="ratplt.pdf")
    args = p.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    from ..plot import ratplt

    ax = ratplt(args.inp, args.dsr, args.error, args.cutp, args.cuts, args.scale, args.upper)
    ax.figure.savefig(args.output, bbox_inches="tight", dpi=200)
    stdout.write(f"wrote {args.output}\n")
    return 0
