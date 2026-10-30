"""Small command-line tools: radiat, freesurf, synth, convert."""

from __future__ import annotations

import argparse
import sys



from ..inp import convert_legacy, write_inp
from ..radiation import free_surface_corrections, radiation_pattern
from ..synth import synthetic_inp
from .errors import clean_errors


@clean_errors("pyfocmec radiat")
def radiat_main(argv=None, stdout=None):
    stdout = stdout or sys.stdout
    p = argparse.ArgumentParser(prog="pyfocmec radiat",
                                description="P, SV and SH radiation factors of a double couple.")
    p.add_argument("--dsr", nargs=3, type=float, required=True, metavar=("DIP", "STRIKE", "RAKE"))
    p.add_argument("--azimuth", nargs="+", type=float, required=True, help="station azimuth(s), degrees")
    p.add_argument("--toa", nargs=2, type=float, required=True, metavar=("P", "S"),
                   help="P and S take-off angles from down, degrees")
    args = p.parse_args(argv)
    stdout.write("Sign conventions: P (up), SV (toward station), SH (left, back to event) are +\n")
    stdout.write("   Dip   Strike    Rake  Azimuth   P TOA   S TOA  P factor      SV     SH   SH/P   SV/P  SV/SH\n")
    for az in args.azimuth:
        pr, svr, shr = (float(x) for x in radiation_pattern(*args.dsr, az, *args.toa))
        row = "".join(f"{v:9.3f}" for v in (*args.dsr, az, *args.toa, pr)) + f"{svr:7.3f}{shr:7.2f}"
        if abs(pr) > 1e-9:
            row += f"   {shr / pr:.4f} {svr / pr:.4f}" + (f" {svr / shr:.4f}" if abs(shr) > 1e-9 else "")
        stdout.write(row + "\n")
    return 0


@clean_errors("pyfocmec freesurf")
def freesurf_main(argv=None, stdout=None):
    stdout = stdout or sys.stdout
    p = argparse.ArgumentParser(prog="pyfocmec freesurf",
                                description="Free-surface amplitude corrections for ratio data.")
    p.add_argument("--vs-vp", type=float, required=True, help="Vs/Vp at the surface")
    p.add_argument("--p-emergence", type=float, required=True, help="P emergence angle from vertical, degrees")
    p.add_argument("--s-emergence", type=float, required=True, help="S emergence angle from vertical, degrees")
    args = p.parse_args(argv)
    res = free_surface_corrections(args.vs_vp, args.p_emergence, args.s_emergence)
    stdout.write(f"Vs/Vp = {args.vs_vp}; critical angle {res['critical_angle']:.3f} degrees\n")
    if res["supercritical"]:
        stdout.write("WARNING: SV emergence at or above critical angle\n")
    for k, v in res.items():
        if k not in ("critical_angle", "supercritical"):
            stdout.write(f"{k:>10s} = {v: .5f}\n")
    return 0


@clean_errors("pyfocmec synth")
def synth_main(argv=None, stdout=None):
    stdout = stdout or sys.stdout
    p = argparse.ArgumentParser(prog="pyfocmec synth",
                                description="Write a synthetic FOCMEC input file for a known mechanism.")
    p.add_argument("--dsr", nargs=3, type=float, required=True, metavar=("DIP", "STRIKE", "RAKE"))
    p.add_argument("--toa", nargs=2, type=float, required=True, metavar=("P", "S"))
    p.add_argument("--vpvs", type=float, default=1.7321)
    p.add_argument("-n", "--stations", type=int, default=15, help="number of azimuths [15]")
    p.add_argument("-o", "--out", default="synth_focmec.inp")
    args = p.parse_args(argv)
    comment, lines = synthetic_inp(*args.dsr, *args.toa, vpvs=args.vpvs, n_azimuths=args.stations)
    write_inp(args.out, comment, lines)
    stdout.write(f"wrote {args.out} ({len(lines)} data lines)\n")
    return 0


@clean_errors("pyfocmec convert")
def convert_main(argv=None, stdout=None):
    stdout = stdout or sys.stdout
    p = argparse.ArgumentParser(prog="pyfocmec convert",
                                description="Convert a pre-2017 FOCMEC input file (adds the Vp/Vs field).")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--vpvs", type=float, default=1.73205, help="Vp/Vs at the source [1.73205]")
    args = p.parse_args(argv)
    with open(args.input) as fh:
        lines = fh.read().splitlines()
    out = convert_legacy(lines, args.vpvs)
    with open(args.output, "w") as fh:
        fh.write("\n".join(out) + "\n")
    return 0
