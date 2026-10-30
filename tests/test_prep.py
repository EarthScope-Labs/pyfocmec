import subprocess
import sys

import pytest

from conftest import FIXTURES
from pyfocmec import prep

PREP = FIXTURES / "prep"


def test_focmec_prep_matches_fortran_short_lines():
    # lines short enough to avoid the 80-character buffer overflow in the Fortran program
    table = prep.read_ray_table(PREP / "focmec_prep_8sta.inp")
    comment, lines, _ = prep.focmec_prep(table, (PREP / "short.dat").read_text().splitlines())
    ref = (PREP / "orig_short.inp").read_text().splitlines()
    assert [comment] + lines == ref


def test_focmec_prep_uses_both_q_terms():
    """The original drops the P attenuation term when the line is long enough to overflow."""
    table = prep.read_ray_table(PREP / "focmec_prep_8sta.inp")
    picks = ["c", "bla  H R -119100.0,-18450.0,500,1000,0.03,0.03"]
    _, lines, log = prep.focmec_prep(table, picks)
    row = table.rows[0]
    import math

    q = math.exp(math.pi * row.ts * 0.03 / 500 - math.pi * row.tp * 0.03 / 1000)
    assert any(f"{q:.4f}" in ln or f"{q:.3f}" in ln for ln in log)
    assert lines[0].startswith("BLA    32.62   31.00H")  # azimuth of BLA, not a corrupted one


def test_halfspace_ratio_prep_matches_fortran():
    vals, stations = prep.read_hypocenter_file(PREP / "hypo1.txt")
    locs = prep.read_station_locations(PREP / "stations_el.loc")
    table = prep.ratio_prep(*vals[:3], stations, locs, prep.HalfSpace(*vals[3:5]), "Half space test")
    ref = prep.read_ray_table(PREP / "orig_table1.txt")
    assert table.vsvp_surface == pytest.approx(ref.vsvp_surface, abs=1e-4)
    for a, b in zip(table.rows, ref.rows):
        assert a.sta == b.sta
        for f in ("dist", "az", "ptoa", "stoa", "pemrg", "semrg"):
            assert getattr(a, f) == pytest.approx(getattr(b, f), abs=2e-3)
        assert a.tp == pytest.approx(b.tp, abs=2e-3)
        assert a.ts == pytest.approx(b.ts, abs=2e-3)


def test_ray_table_roundtrip(tmp_path):
    ref = prep.read_ray_table(PREP / "orig_table1.txt")
    prep.write_ray_table(ref, tmp_path / "t.txt")
    again = prep.read_ray_table(tmp_path / "t.txt")
    assert [r.sta for r in again.rows] == [r.sta for r in ref.rows]
    assert again.rows[0].dist == pytest.approx(ref.rows[0].dist, abs=1e-4)


def test_distaz_known_value():
    d, az, baz = prep.distaz(49.024, 141.937, 37.2111, -80.42054)
    assert d == pytest.approx(86.2757, abs=1e-3)
    assert az == pytest.approx(32.619, abs=1e-2)


def test_hypo2foc_matches_fortran(tmp_path):
    files = prep.hypo2foc(PREP / "8703270729.arc", "Chapman et al. 1997 Event 12: 8703270729305.  Emergent coded",
                          tmp_path)
    assert [f.name for f in files] == ["8703270729.inp"]
    assert files[0].read_text() == (FIXTURES / "inputs" / "8703270729.inp").read_text()


def test_prep_cli_roundtrip(tmp_path):
    out = tmp_path / "x.inp"
    p = subprocess.run([sys.executable, "-m", "pyfocmec.cli.main", "prep", str(PREP / "focmec_prep_8sta.inp"),
                        str(PREP / "short.dat"), "-o", str(out), "--float32"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert out.read_text() == (PREP / "orig_short.inp").read_text()
