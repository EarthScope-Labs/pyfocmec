"""Commands that prepare FOCMEC input: ratio-prep, prep (focmec_prep) and hypo2foc."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .. import prep
from .errors import clean_errors


@clean_errors("pyfocmec ratio-prep")
def ratio_prep_main(argv=None, stdout=None):
    stdout = stdout or sys.stdout
    p = argparse.ArgumentParser(
        prog="pyfocmec ratio-prep",
        description="Take-off/emergence angles and travel times for a hypocentre and stations "
        "(ratio_prep).  The first line of HYPOCENTRE_FILE is 'lat lon depth_km' (plus 'vp vs' "
        "for a half-space), followed by one station name per line.",
    )
    p.add_argument("hypocenter_file")
    p.add_argument("--stations", default="stations.loc", help="station locations file [stations.loc]")
    p.add_argument("--model", default=None,
                   help="halfspace, or a TauP model: iasp91, ak135, nocrust or a .npz path "
                        "(default: halfspace if the file gives vp and vs, else ak135)")
    p.add_argument("--vp", type=float, help="half-space P velocity, km/s")
    p.add_argument("--vs", type=float, help="half-space S velocity, km/s")
    p.add_argument("--no-elevation", action="store_true", help="ignore station elevations (half-space)")
    p.add_argument("-c", "--comment", default="", help="comment line for the table")
    p.add_argument("-o", "--out", default="focmec_prep_input.txt")
    args = p.parse_args(argv)
    vals, stations = prep.read_hypocenter_file(args.hypocenter_file)
    lat, lon, depth = vals[:3]
    vp = args.vp if args.vp else (vals[3] if len(vals) >= 5 else None)
    vs = args.vs if args.vs else (vals[4] if len(vals) >= 5 else None)
    model = args.model or ("halfspace" if vp and vs else "ak135")
    if model == "halfspace":
        if not (vp and vs):
            p.error("a half-space needs vp and vs (in the hypocentre file or --vp/--vs)")
        backend = prep.HalfSpace(vp, vs)
    else:
        backend = prep.TauP(model)
    locs = prep.read_station_locations(args.stations)
    table = prep.ratio_prep(lat, lon, depth, stations, locs, backend, args.comment,
                            use_elevation=not args.no_elevation)
    prep.write_ray_table(table, args.out)
    stdout.write(f"wrote {args.out}: {len(table.rows)} stations, Vs/Vp surface {table.vsvp_surface:.4f}, "
                 f"Vp/Vs focus {table.vpvs_focus:.4f}\n")
    return 0


@clean_errors("pyfocmec prep")
def focmec_prep_main(argv=None, stdout=None):
    stdout = stdout or sys.stdout
    p = argparse.ArgumentParser(
        prog="pyfocmec prep",
        description="Combine picks with a ratio-prep table to make a FOCMEC input file (focmec_prep).",
    )
    p.add_argument("table", help="file written by `pyfocmec ratio-prep`")
    p.add_argument("picks", help="picks file: comment line, then 'STA K' or 'STA K N num,den[,Qs,Qp,fs,fp]'")
    p.add_argument("-o", "--out", default="focmec.inp")
    p.add_argument("--log", help="write a processing log (like focmec_prep.lst)")
    p.add_argument("--float32", action="store_true", help="single precision like the original")
    args = p.parse_args(argv)
    import numpy as np

    table = prep.read_ray_table(args.table)
    try:
        comment, lines, log = prep.focmec_prep(table, Path(args.picks).read_text().splitlines(),
                                               np.float32 if args.float32 else np.float64)
    except prep.PrepError as exc:
        print(f"pyfocmec prep: {exc}", file=sys.stderr)
        return 1
    Path(args.out).write_text("\n".join([comment] + lines) + "\n")
    if args.log:
        Path(args.log).write_text("\n".join(log) + "\n")
    stdout.write(f"wrote {args.out}: {len(lines)} data lines\n")
    return 0


@clean_errors("pyfocmec hypo2foc")
def hypo2foc_main(argv=None, stdout=None):
    stdout = stdout or sys.stdout
    p = argparse.ArgumentParser(prog="pyfocmec hypo2foc",
                                description="Convert a hypoellipse archive file to FOCMEC input files.")
    p.add_argument("archive")
    p.add_argument("-c", "--comment", default="", help="comment line for every event")
    p.add_argument("--outdir", default=".")
    args = p.parse_args(argv)
    for f in prep.hypo2foc(args.archive, args.comment, args.outdir):
        stdout.write(f"wrote {f}\n")
    return 0
