"""Preparing FOCMEC input: ``ratio_prep``, ``focmec_prep`` and ``hypo2foc``.

Workflow for data with amplitude ratios::

    ratio_prep(hypocentre, stations)  ->  RayTable      (angles, travel times)
    focmec_prep(RayTable, picks)      ->  FOCMEC .inp   (polarities + free-surface/Q-corrected ratios)

For polarity-only data :func:`hypo2foc` converts a hypoellipse archive file.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .fortfmt import fnum, rec
from .radiation import critical_angle, free_surface

_B = 0.9932773  # geocentric latitude factor used by the original disaz


# ---------------------------------------------------------------- geometry
def distaz(elat, elon, slat, slon):
    """Distance, azimuth (event to station) and back azimuth in degrees (port of ``disaz``).

    Uses the original's ellipticity correction (geocentric latitude with
    b/a = 0.9932773).  Inputs are degrees; like the Fortran, inputs are held
    in single precision.
    """
    f32 = lambda v: float(np.float32(v))  # noqa: E731
    elat, elon, slat, slon = (f32(v) for v in (elat, elon, slat, slon))
    rd = 180.0 / math.pi
    elat_r, elon_r, slat_r, slon_r = (v / rd for v in (elat, elon, slat, slon))
    pi2 = math.pi / 2
    twopi = 2 * math.pi
    if elat_r == slat_r and elon_r == slon_r:
        return 0.0, 0.0, 180.0
    ela = pi2 - math.atan2(_B * math.sin(elat_r), math.cos(elat_r))
    sla = pi2 - math.atan2(_B * math.sin(slat_r), math.cos(slat_r))
    elon_ = elon_r + twopi if elon_r < 0 else elon_r
    slon_ = slon_r + twopi if slon_r < 0 else slon_r
    eslon = elon_ - slon_
    cosela, sinela = math.cos(ela), math.sin(ela)
    cossla, sinsla = math.cos(sla), math.sin(sla)
    cosdel = cosela * cossla + sinela * sinsla * math.cos(eslon)
    dist = math.acos(max(-1.0, min(1.0, cosdel)))
    sindel = math.sin(dist)
    if sinela == 0.0 or sinsla == 0.0 or sindel == 0.0:
        # the original divides by these and returns NaN for an event at a pole or an antipodal
        # station; the same azimuths written without the division stay finite
        aze = math.atan2(-math.sin(eslon) * sinsla, sinela * cossla - cosela * sinsla * math.cos(eslon))
        azs = math.atan2(math.sin(eslon) * sinela, sinsla * cosela - cossla * sinela * math.cos(eslon))
    else:
        factor = -math.sin(eslon) / sindel
        aze = math.atan2(factor * sinsla, (cossla - cosela * cosdel) / (sinela * sindel))
        azs = math.atan2(-factor * sinela, (cosela - cossla * cosdel) / (sinsla * sindel))
    aze = aze + twopi if aze < 0 else aze + 0.0  # + 0.0 turns -0.0 into 0.0
    azs = azs + twopi if azs < 0 else azs + 0.0
    return dist * rd, aze * rd, azs * rd


# ---------------------------------------------------------------- station locations
def read_station_locations(path) -> dict:
    """``stations.loc``: ``NAME lat lon [elevation_m]`` per line (free format).

    Like the original, only the numbers after the name are read (text after them, such as a
    station description, is ignored) and the first line for a name wins.
    """
    locs = {}
    for line in Path(path).read_text().splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) < 3:
            continue
        vals = []
        for tok in parts[1:4]:
            try:
                vals.append(float(tok))
            except ValueError:
                break
        if len(vals) < 2:
            continue
        locs.setdefault(parts[0].upper(), tuple(vals))
    return locs


# ---------------------------------------------------------------- ray tables
@dataclass
class RayRow:
    sta: str
    dist: float
    az: float
    ptoa: float
    stoa: float
    pemrg: float
    semrg: float
    tp: float
    ts: float


@dataclass
class RayTable:
    """Per-station angles and travel times (the file ``ratio_prep`` writes)."""

    lat: float
    lon: float
    depth: float
    comment: str
    vsvp_surface: float
    vpvs_focus: float
    rows: list = field(default_factory=list)
    layout: str = "model"  # "model" (tau-p) or "halfspace" column widths


def write_ray_table(table: RayTable, path) -> None:
    """Write the ``focmec_prep`` input file in the original column layout."""
    lines = [
        f"{fnum(table.lat, 8, 4)}{fnum(table.lon, 10, 4)}{fnum(table.depth, 8, 3)}  {table.comment}",
        f"{fnum(table.vsvp_surface, 6, 4)} {fnum(table.vpvs_focus, 6, 4)} S/P surface and P/S focus velocity ratios",
        rec(1, "Stn", 8, "Dist", 17, "Az", 26, "Ptoa", 35, "Stoa", 44, "Pemrg", 53, "Semrg", 62, "TT(P)", 71, "TT(S)"),
    ]
    for r in table.rows:
        sta = r.sta[:4].ljust(4)
        if table.layout == "halfspace":
            body = fnum(r.dist, 10, 5) + fnum(r.az, 8, 3)
            rest = (r.ptoa, r.stoa, r.pemrg, r.semrg, r.tp, r.ts)
        else:
            body = fnum(r.dist, 9, 4)
            rest = (r.az, r.ptoa, r.stoa, r.pemrg, r.semrg, r.tp, r.ts)
        lines.append(sta + body + "".join(fnum(v, 9, 3) for v in rest))
    Path(path).write_text("\n".join(lines) + "\n")


def read_ray_table(path) -> RayTable:
    """Read a ray table written by ``ratio_prep`` (either column layout)."""
    lines = Path(path).read_text().splitlines()
    head = lines[0]
    m = re.match(r"\s*(\S+)\s+(\S+)\s+(\S+)\s?(.*)", head)
    try:
        lat, lon, dep = float(m.group(1)), float(m.group(2)), float(m.group(3))
        comment = m.group(4).strip()
    except (AttributeError, ValueError):
        # focmec_prep.f treats line 1 as a free comment (older or hand-made tables have no numbers)
        lat = lon = dep = float("nan")
        comment = head.strip()
    vs, vp = (float(x) for x in lines[1].split()[:2])
    rows = []
    for ln in lines[3:]:
        if not ln.strip():
            continue
        vals = [float(x) for x in ln[4:].split()]
        rows.append(RayRow(ln[:4].strip().upper(), *vals[:8]))
    return RayTable(lat, lon, dep, comment, vs, vp, rows)


class HalfSpace:
    """Straight rays in a uniform half-space (``ratio_prep_1layer``)."""

    layout = "halfspace"

    def __init__(self, vp: float, vs: float):
        self.vp, self.vs = float(vp), float(vs)

    @property
    def vsvp_surface(self):
        return self.vs / self.vp

    def vpvs_focus(self, depth):
        return self.vp / self.vs

    def rays(self, dist_deg, depth, elevation=0.0):
        rd = 180.0 / math.pi
        edistkm = 111.195 * dist_deg
        totdep = depth + elevation
        dist = math.hypot(edistkm, totdep)
        pemrg = rd * math.acos(totdep / dist)
        toang = 180.0 - pemrg
        return dict(ptoa=toang, stoa=toang, pemrg=pemrg, semrg=pemrg, tp=dist / self.vp, ts=dist / self.vs)


class TauP:
    """Rays in a 1-D Earth model through ``obspy.taup`` (``pip install pyfocmec[taup]``).

    ``model`` is a built-in name (``"iasp91"``, ``"ak135"``), ``"nocrust"`` (ak135
    without the crust, shipped with pyfocmec) or the path of a ``.npz`` model.
    """

    layout = "model"

    def __init__(self, model="ak135"):
        try:
            from obspy.taup import TauPyModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError("the TauP backend needs obspy: pip install pyfocmec[taup]") from exc
        self.name = str(model)
        path = _model_path(model)
        self.model = TauPyModel(model=path)
        vmod = self.model.model.s_mod.v_mod
        self._vmod = vmod
        self.vp_surface = float(np.squeeze(vmod.evaluate_below(0.0, "P")))
        self.vs_surface = float(np.squeeze(vmod.evaluate_below(0.0, "S")))

    @property
    def vsvp_surface(self):
        return self.vs_surface / self.vp_surface

    def vpvs_focus(self, depth):
        return float(np.squeeze(self._vmod.evaluate_below(depth, "P")) / np.squeeze(self._vmod.evaluate_below(depth, "S")))

    def rays(self, dist_deg, depth, elevation=0.0):
        # like the original, only direct P and S arrivals (no depth or converted phases); upgoing
        # (p, s), regional (Pn, Pg, Sn, Sg) and diffracted/core phases all count, as in ttimes
        arr = self.model.get_travel_times(source_depth_in_km=depth, distance_in_degree=dist_deg,
                                          phase_list=["ttbasic"])
        parr = min((a for a in arr if _P_PHASES.match(a.name)), key=lambda a: a.time, default=None)
        sarr = min((a for a in arr if _S_PHASES.match(a.name)), key=lambda a: a.time, default=None)
        if parr is None or sarr is None:
            raise PrepError(
                f"no direct {'P' if parr is None else 'S'} arrival in model {self.name} "
                f"at distance {dist_deg:.3f} deg, depth {depth:g} km"
            )
        return dict(ptoa=parr.takeoff_angle, stoa=sarr.takeoff_angle, pemrg=parr.incident_angle,
                    semrg=sarr.incident_angle, tp=parr.time, ts=sarr.time)


_P_PHASES = re.compile(r"^(p|P|Pn|Pg|Pb|P\*|Pdiff|PKP|PKiKP|PKIKP|PKPab|PKPbc|PKPdf|PKPdiff)$")
_S_PHASES = re.compile(r"^(s|S|Sn|Sg|Sb|S\*|Sdiff)$")


def _model_path(model):
    """Built-in model names pass through; ``nocrust`` is built from the shipped ``.tvel``."""
    if str(model).lower() != "nocrust":
        return str(model)
    import os
    from importlib import resources

    import obspy

    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pyfocmec" / f"obspy-{obspy.__version__}"
    target = cache / "nocrust.npz"
    if not target.exists():
        from obspy.taup.taup_create import build_taup_model

        cache.mkdir(parents=True, exist_ok=True)
        tvel = resources.files("pyfocmec") / "data" / "nocrust.tvel"
        with resources.as_file(tvel) as f:
            build_taup_model(str(f), output_folder=str(cache))
    return str(target)


def read_hypocenter_file(path):
    """``lat lon depth [vp vs]`` on the first line, then station names (one per line)."""
    lines = Path(path).read_text().splitlines()
    vals = [float(x) for x in lines[0].replace(",", " ").split()]
    stations = [ln.strip() for ln in lines[1:] if ln.strip()]
    return vals, stations


def ratio_prep(lat, lon, depth, stations, locations, backend, comment="", use_elevation=None):
    """Angles and travel times for every station (port of ``ratio_prep``/``ratio_prep_1layer``).

    ``stations`` are names; ``locations`` maps names to ``(lat, lon[, elevation_m])``
    (see :func:`read_station_locations`).  ``backend`` is :class:`HalfSpace` or :class:`TauP`.
    Station elevations (metres) are only used by the half-space backend.
    """
    rows = []
    halfspace = isinstance(backend, HalfSpace)
    for name in stations:
        key = name.upper()
        if key not in locations:
            raise KeyError(f"No station called {name} listed")
        loc = locations[key]
        d, az, _ = distaz(lat, lon, loc[0], loc[1])
        elev = (loc[2] / 1000.0 if len(loc) > 2 else 0.0) if (halfspace and use_elevation is not False) else 0.0
        r = backend.rays(d, depth, elev)
        rows.append(RayRow(key[:4].strip(), d, az, r["ptoa"], r["stoa"], r["pemrg"], r["semrg"], r["tp"], r["ts"]))
    return RayTable(
        float(np.float32(lat)), float(np.float32(lon)), float(depth), comment,
        backend.vsvp_surface, backend.vpvs_focus(depth), rows, backend.layout,
    )


# ---------------------------------------------------------------- focmec_prep
class PrepError(ValueError):
    pass


def _put(buf, col, text):
    """Write ``text`` at 1-based ``col`` of a fixed-width character buffer."""
    for i, ch in enumerate(text):
        if col - 1 + i < len(buf):
            buf[col - 1 + i] = ch


def _get(buf, c1, c2):
    return "".join(buf[c1 - 1:c2])


def _lf_float(text: str) -> float:
    """First token of a Fortran list-directed read: ignores trailing text, accepts ``d`` exponents."""
    toks = text.split()
    if not toks:
        raise PrepError(f"cannot read a number from {text!r}")
    try:
        return float(toks[0].replace("d", "e").replace("D", "E"))
    except ValueError:
        raise PrepError(f"cannot read a number from {text!r}") from None


def _ratio_value(buf, lc, row, vsvp, log, dtype):
    """Port of ``getrat``: observed ratio -> log10 corrected ratio."""
    dt = np.dtype(dtype).type
    sense = buf[20]
    text = _get(buf, 46, lc)
    parts = text.split(",")
    if len(parts) == 1:
        raise PrepError(f"No delimiter in {_get(buf, 40, lc)}")
    if len(parts) == 2:
        anum, aden = (dt(_lf_float(p)) for p in parts)
        qfact = dt(1.0)
    elif len(parts) >= 6:  # like the original, anything after the sixth value is ignored
        anum, aden, qs, qp, fs, fp = (dt(_lf_float(p)) for p in parts[:6])
        pi = dt(math.pi)
        # Deliberate divergence: focmec_prep.f always uses the P travel time in the second term.
        # For an SV/SH ratio both waves are S waves, so the S travel time belongs there.
        t_den = row.ts if sense == "S" else row.tp
        qfact = np.exp(pi * dt(row.ts) * fs / qs - pi * dt(t_den) * fp / qp)
        log.append(f" Q correction: {float(qfact)}")
    else:
        raise PrepError(f"Not enough delimiters in {_get(buf, 40, lc)}")
    obrat = anum / aden
    sar, sav = free_surface(2, vsvp, row.semrg, dtype)[:2]
    angcrit = critical_angle(vsvp)
    sar, sav = dt(sar), dt(sav)
    if sense == "S":
        fsfact = dt(2.0) / sar
    else:
        par, pav = (dt(x) for x in free_surface(1, vsvp, row.pemrg, dtype)[:2])
        fsfact = None
        if sense == "V":
            fsfact = pav / sar
            if row.semrg >= angcrit:
                log.append(f" WARNING SV emerge and critical angle: {row.semrg} {angcrit}")
                log.append(f" Sar/Sav = {sar / sav}")
        elif sense == "H":
            fsfact = pav / dt(2.0)
    if sense == "S" and row.semrg >= angcrit:
        log.append(f" WARNING SV emerge and critical angle: {row.semrg} {angcrit}")
    ratl10 = np.log10(abs(fsfact * qfact * obrat))
    _put(buf, 22, fnum(ratl10, 8, 4))
    log.append(f" Observed ratio: {obrat} Free-surface: {fsfact} Q correction: {qfact}")


def focmec_prep(table: RayTable, picks, dtype=np.float64):
    """Make a FOCMEC input file from picks and a ray table (port of ``focmec_prep``).

    ``picks`` is an iterable of lines whose first line is the comment.  Data lines
    are ``STA K`` for polarities (``K`` the polarity key) and
    ``STA K N amp_num,amp_den[,Qs,Qp,fs,fp]`` for ratios, where ``K`` is ``V``
    (SV/P), ``H`` (SH/P) or ``S`` (SV/SH), ``N`` the polarity of the numerator
    and the amplitudes are the observed numerator and denominator.  With the
    four optional values (Q of S and P, and S and P frequencies) the ratios are
    also corrected for attenuation.

    Returns ``(comment, lines, log)``.
    """
    lines = [ln.rstrip("\n") for ln in picks]
    if not lines:
        raise PrepError("empty picks file")
    comment = lines[0].rstrip()
    by_sta = {}
    for r in table.rows:  # the original uses the first row listed for a station
        by_sta.setdefault(r.sta.strip().upper(), r)
    out, log = [], [f" {comment}"]
    for raw in lines[1:]:
        raw = raw.rstrip()
        if not raw.strip():
            continue
        log.append(f" Input: {raw}")
        sta = raw[:4].upper().ljust(4)
        row = by_sta.get(sta.strip())
        if row is None:
            raise PrepError(f"No station {sta.strip()} in list")
        buf = list(raw.ljust(200))
        lc = len(raw)
        sense = buf[5]
        if sense in "VHS":
            if lc <= 6:
                raise PrepError(f"No ratio info in line {raw}")
            numpol = buf[7]
            amp = _get(buf, 10, lc)
            buf[:] = list(" " * 200)
            _put(buf, 1, sta)
            _put(buf, 30, " " + numpol)
            _put(buf, 40, fnum(table.vpvs_focus, 6, 4))
            _put(buf, 47, amp)
            lc = 46 + len(amp)
            if lc > len(buf):
                raise PrepError(f"amplitude text too long ({len(amp)} characters) in line {raw[:20]}...")
            toang1 = row.stoa
            toang2 = row.ptoa if sense in "VH" else toang1
            _put(buf, 5, fnum(row.az, 8, 2))
            _put(buf, 21, sense)
            _put(buf, 32, fnum(toang2, 7, 2))
            _put(buf, 46, " ")
            _ratio_value(buf, lc, row, table.vsvp_surface, log, dtype)
        else:
            comment_text = raw[6:].strip()
            buf[:] = list(" " * 200)
            _put(buf, 1, sta)
            _put(buf, 5, fnum(row.az, 8, 2))
            buf[20] = sense
            lc = 21
            if sense in "CUD+-e":
                toang1 = row.ptoa
            elif sense in "<>FBlru" or sense in "LR":
                toang1 = row.stoa
            else:
                raise PrepError(f"{sense} is not a legal option")
            if comment_text:
                _put(buf, 40, comment_text)
                lc = 39 + len(comment_text)
        _put(buf, 13, fnum(toang1, 8, 2))
        text = "".join(buf[:lc]).rstrip() if lc > 21 else "".join(buf[:lc])
        out.append(text)
        log.append(f" {text}")
    return comment, out, log


# ---------------------------------------------------------------- hypo2foc
def _f3(field: str, what: str, sta: str) -> float:
    """``F3.0`` field: blank is zero, each field on its own."""
    if not field.strip():
        return 0.0
    try:
        return float(field)
    except ValueError:
        raise PrepError(f"cannot read {what} {field!r} for station {sta.strip()}") from None


def hypo2foc(path, comment="", outdir=".", comments=None):
    """Convert a hypoellipse archive file to FOCMEC polarity input files.

    One ``YYMMDDHHMM.inp`` file is written per event (named from the first ten
    characters of the summary record).  ``comment`` is the comment line used for
    every event unless ``comments`` (a callable taking the event's name) is given.
    Returns the list of files written.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    lines = Path(path).read_text().splitlines()
    written = []
    i = 0
    while i < len(lines):
        head = lines[i].ljust(115)
        i += 1
        if not head.strip():
            continue  # a blank line (e.g. at the end of the file) is not an event
        name = "".join("0" if c == " " else c for c in head[:10]) + ".inp"
        text = comments(name) if callable(comments) else comment
        body = [text.rstrip()]
        while i < len(lines):
            line = lines[i].ljust(115)
            i += 1
            if line[:4] == "    ":
                break
            sta, des, pol, weight = line[0:4], line[4:6], line[6], line[7]
            azm = _f3(line[28:31], "azimuth", sta)
            ain = _f3(line[40:43], "incidence angle", sta)
            sense = pol
            if pol in "cu":
                sense = "C"
            if pol == "d":
                sense = "D"
            if sense == "C" and des == "eP":
                sense = "+"
            if sense == "D" and des == "eP":
                sense = "-"
            if pol != " ":
                body.append(rec(1, sta, fnum(azm, 8, 2), fnum(ain, 8, 2), sense, 40, des + " " + weight))
        target = outdir / name
        target.write_text("\n".join(body) + "\n")
        written.append(target)
    return written
