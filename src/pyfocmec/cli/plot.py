"""Plotting commands: focplt and ratplt."""

from __future__ import annotations

import argparse
import sys

from .errors import clean_errors

# line styles of successive solution sets: dash length and gap in inches, as in the original sample runs
STYLES = {"solid": None, "dash": (0.15, 0.02), "dot": (0.02, 0.02), "long": (0.3, 0.05)}


@clean_errors("pyfocmec focplt")
def focplt_main(argv=None, stdout=None):
    stdout = stdout or sys.stdout
    p = argparse.ArgumentParser(prog="pyfocmec focplt", description="Plot polarities and focal mechanism solutions.")
    p.add_argument("--inp", help="FOCMEC input file (polarities)")
    p.add_argument("--out", action="append", default=[], metavar="FILE",
                   help="focmec.out file with solutions (repeat for several sets)")
    p.add_argument("--dsr", action="append", default=[], nargs=3, type=float, metavar=("DIP", "STRIKE", "RAKE"),
                   help="one mechanism (repeat for several)")
    p.add_argument("--styles", nargs="+", default=["solid", "dash", "dot"], choices=sorted(STYLES),
                   help="line style of the 1st, 2nd, ... solution set (the --out files, then the --dsr); "
                        "the last one repeats [solid dash dot]")
    p.add_argument("--surface", choices=["P", "SH", "SV"], default="P",
                   help="draw the P nodal planes or the SH or SV nodal surfaces [P]")
    p.add_argument("--axes", nargs="+", default=[], choices=["P", "T", "B"], help="mark these axes of the --dsr")
    p.add_argument("--linewidth", type=float, default=1.0, help="pen number of the solutions [1]")
    p.add_argument("--upper", action="store_true", help="upper hemisphere")
    p.add_argument("--phases", nargs="+", default=["P", "SV", "SH"], choices=["P", "SV", "SH"])
    p.add_argument("--emergent", action="store_true", help="also plot emergent picks")
    p.add_argument("--errors", action="store_true", help="box the picks flagged with an E line")
    p.add_argument("--size", type=float, default=0.25, help="size of the impulsive symbols, inches of the original [0.25]")
    p.add_argument("--emergent-size", type=float, default=0.15, help="size of the emergent symbols [0.15]")
    p.add_argument("--nodal-surfaces", action="store_true", help="also the SV and SH nodal surfaces of each --dsr")
    p.add_argument("--beachball", action="store_true", help="shade the compressional quadrants of the first --dsr")
    p.add_argument("--labels", action="store_true", help="label stations")
    p.add_argument("--label", help="plot label such as 'a)', top left")
    p.add_argument("--caption", action="store_true", help="add a 'lower hemisphere' caption")
    p.add_argument("--title")
    p.add_argument("-o", "--output", default="focplt.pdf", help="image file (.pdf, .png, .svg, .eps) [focplt.pdf]")
    args = p.parse_args(argv)
    if not (args.inp or args.out or args.dsr):
        p.error("give --inp, --out and/or --dsr")
    import matplotlib

    matplotlib.use("Agg")
    from ..plot import focplt

    sets = []
    sources = [{"solutions": f} for f in args.out] + [{"dsr": d} for d in args.dsr]
    for k, src in enumerate(sources):
        style = STYLES[args.styles[min(k, len(args.styles) - 1)]]
        s = dict(src, surface=args.surface, dash=style, lw=args.linewidth)
        if "dsr" in src:
            s["axes"] = tuple(args.axes)
        sets.append(s)
        if "dsr" in src and args.nodal_surfaces:
            sets += [dict(dsr=src["dsr"], surface="SV", dash=STYLES["dash"]),
                     dict(dsr=src["dsr"], surface="SH", dash=STYLES["dot"])]
    ax = focplt(args.inp, sets=sets, dsr=None, upper=args.upper, phases=tuple(args.phases), emergent=args.emergent,
                title=args.title, labels=args.labels, label=args.label, size=args.size,
                emergent_size=args.emergent_size, errors=args.errors, caption=args.caption)
    if args.beachball and args.dsr:
        from ..plot import plot_beachball

        plot_beachball(args.dsr[0], ax=ax, upper=args.upper)
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
    p.add_argument("--scale", type=float, default=0.20, help="symbol size for |o-c| = 0, inches of the original [0.2]")
    p.add_argument("--upper", action="store_true")
    p.add_argument("--title")
    p.add_argument("--label", help="plot label such as 'a)', top left")
    p.add_argument("--legend", action="store_true", help="add a key to the symbols")
    p.add_argument("-o", "--output", default="ratplt.pdf")
    args = p.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    from ..plot import ratplt

    ax = ratplt(args.inp, args.dsr, args.error, args.cutp, args.cuts, args.scale, args.upper, title=args.title,
                label=args.label, legend=args.legend)
    ax.figure.savefig(args.output, bbox_inches="tight", dpi=200)
    stdout.write(f"wrote {args.output}\n")
    return 0
