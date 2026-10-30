"""Deliberate divergences from the Fortran program, and fixes found when reviewing the port.

Where the original has a clear bug the port corrects it (see CHANGELOG.md); these tests pin the
corrected behaviour.  Behaviour that must match the original is covered by the oracle fixtures
in test_regression.py, test_dsretc.py and test_prep.py.
"""

import io
import math
import warnings

import numpy as np
import pytest

from conftest import FIXTURES
from pyfocmec import FocmecParams, dsretc as ds, geometry, parse_inp, prep, search
from pyfocmec.fortfmt import fnum
from pyfocmec.inp import InputError, has_emergent_picks
from pyfocmec.prompts import Prompter

PREP = FIXTURES / "prep"


# ------------------------------------------------------------------ focmec_prep Q correction
def _ratio_field(line):
    return float(line[21:29])


def _prep_lines(picks):
    table = prep.read_ray_table(PREP / "focmec_prep_8sta.inp")
    return prep.focmec_prep(table, ["c"] + picks)[1]


def test_q_correction_sv_sh_uses_s_travel_time_for_both_terms():
    # SV/SH: both waves are S waves.  With equal Q and frequency the correction is exactly 1,
    # so the Q-corrected ratio must equal the uncorrected one.
    plain, with_q = _prep_lines(["bla  S F 10,20", "bla  S F 10,20,500,500,1,1"])
    assert _ratio_field(with_q) == _ratio_field(plain)


def test_q_correction_sv_p_still_uses_both_travel_times():
    # SV/P: S and P travel times differ, so equal Q and frequency still give a correction
    plain, with_q = _prep_lines(["bla  V F 10,20", "bla  V F 10,20,500,500,1,1"])
    row = prep.read_ray_table(PREP / "focmec_prep_8sta.inp").rows[0]
    expected = math.pi * (row.ts - row.tp) / 500 / math.log(10)
    assert _ratio_field(with_q) - _ratio_field(plain) == pytest.approx(expected, abs=2e-4)


# ------------------------------------------------------------------ focmec_prep input parsing
@pytest.mark.parametrize(
    "amp",
    ["10,20 measured on Z", "1d1,2d1", "10,20,500,500,1,1,extra,fields", "10,20,500,500,1,1 trailing"],
)
def test_prep_amplitudes_parse_like_fortran_list_directed_reads(amp):
    ref = _prep_lines(["bla  H R 10,20" if amp.startswith(("10,20 ", "1d1")) else "bla  H R 10,20,500,500,1,1"])
    got = _prep_lines([f"bla  H R {amp}"])
    assert _ratio_field(got[0]) == _ratio_field(ref[0])


def test_prep_amplitude_errors_are_prep_errors():
    for bad in ("bla  H R 10", "bla  H R 10,20,500", "bla  H R abc,def"):
        with pytest.raises(prep.PrepError):
            _prep_lines([bad])


def test_prep_amplitude_text_too_long_is_an_error():
    with pytest.raises(prep.PrepError, match="too long"):
        _prep_lines(["bla  H R 10,20," + "5," * 100 + "1"])


def test_ray_table_first_line_may_be_a_free_comment(tmp_path):
    text = (PREP / "focmec_prep_8sta.inp").read_text().splitlines()
    text[0] = "Sakhalin hand-made table"
    (tmp_path / "t.txt").write_text("\n".join(text) + "\n")
    table = prep.read_ray_table(tmp_path / "t.txt")
    assert table.comment == "Sakhalin hand-made table"
    assert len(table.rows) == len(prep.read_ray_table(PREP / "focmec_prep_8sta.inp").rows)


def test_station_locations_ignore_trailing_text_and_first_entry_wins(tmp_path):
    (tmp_path / "s.loc").write_text(
        "BLA 37.2 -80.4 634 Blacksburg, Virginia\nBLA 1.0 2.0\nKEV 69.7 27.0\nbad line here\n"
    )
    locs = prep.read_station_locations(tmp_path / "s.loc")
    assert locs["BLA"] == (37.2, -80.4, 634.0)
    assert locs["KEV"] == (69.7, 27.0)
    assert "BAD" not in locs


def test_ray_table_duplicate_station_uses_first_row(tmp_path):
    table = prep.read_ray_table(PREP / "focmec_prep_8sta.inp")
    dup = prep.RayRow(**{**table.rows[0].__dict__, "az": 123.0})
    table.rows.append(dup)
    line = prep.focmec_prep(table, ["c", "bla  D"])[1][0]
    assert line.startswith("BLA    32.62")  # not 123.00


def test_distaz_is_finite_at_the_pole_and_for_antipodes():
    for args in ((90.0, 0.0, 10.0, 10.0), (0.0, 0.0, 0.0, 180.0), (-90.0, 0.0, -10.0, 10.0)):
        assert all(math.isfinite(v) for v in prep.distaz(*args))
    d, aze, azs = prep.distaz(90.0, 0.0, 10.0, 10.0)
    assert aze == pytest.approx(170.0, abs=1e-6) and azs == pytest.approx(0.0, abs=1e-6)


def test_hypo2foc_ignores_blank_lines_and_zeroes_blank_fields_separately(tmp_path):
    arc = (PREP / "8703270729.arc").read_text().splitlines()
    pick = arc[1].ljust(115)
    arc[1] = pick[:28] + "   " + pick[31:]  # blank azimuth only
    (tmp_path / "x.arc").write_text("\n".join(arc) + "\n\n\n")
    files = prep.hypo2foc(tmp_path / "x.arc", "c", tmp_path)
    assert [f.name for f in files] == ["8703270729.inp"]  # no 0000000000.inp from the blank lines
    line = files[0].read_text().splitlines()[1]
    assert line[4:12].strip() == "0.00" and line[12:20].strip() == "148.00"


# ------------------------------------------------------------------ TauP phase selection
def test_taup_finds_direct_phases_at_local_and_regional_distances():
    pytest.importorskip("obspy")
    t = prep.TauP("iasp91")
    for dist, depth in ((0.05, 10.0), (0.5, 10.0), (3.0, 15.0), (30.0, 15.0)):
        r = t.rays(dist, depth)
        assert 0 < r["ptoa"] < 180 and 0 < r["stoa"] < 180
        assert 0 < r["tp"] < r["ts"]
    assert t.rays(0.5, 10.0)["ptoa"] > 90  # upgoing ray from a shallow source


# ------------------------------------------------------------------ input format
def test_emergent_sv_picks_are_used():
    lines = ["t", "AAA    10.00   20.00C", "BBB    20.00   30.00f", "CCC    30.00   40.00b"]
    assert has_emergent_picks(lines)
    d = parse_inp(lines, emergent=True)
    assert list(d.pol_kind) == [0, 1, 1] and list(d.pol_sign) == [1.0, 1.0, -1.0]
    assert parse_inp(lines, emergent=False).npol == 1


def test_emergent_key_on_a_ratio_layout_line_is_still_emergent():
    # focinp.f tests the key in column 21 on every line, whatever else is on it
    lines = ["t", "BBB    90.00   30.00+  0.1234 F  30.00 1.7300"]
    assert has_emergent_picks(lines)
    assert parse_inp(lines, emergent=True).npol == 1 and parse_inp(lines, emergent=False).npol == 0


def test_numbers_without_a_decimal_point_are_read_literally_with_a_warning():
    with pytest.warns(UserWarning, match="no decimal point"):
        d = parse_inp(["t", "AAA      180      45C"])
    ref = parse_inp(["t", "AAA   180.00   45.00C"])
    assert np.allclose(d.pol_xyz, ref.pol_xyz)  # 180 degrees, not the implied-decimal 1.80


def test_blank_inside_a_number_is_a_clear_error():
    with pytest.raises(InputError, match="line 2: cannot read azimuth"):
        parse_inp(["t", "CCC    1 2.5   4 0.0C"])


def test_ratio_line_without_ratio_data_is_an_error():
    with pytest.raises(InputError, match="no ratio data"):
        parse_inp(["t", "AAA    10.00   20.00V"])


def test_no_warning_for_normal_input():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parse_inp(["t", "AAA    10.00   20.00C", "BBB     0.00    0.00D"])


def test_filename_answer_with_blanks_before_a_tab():
    pr = Prompter(io.StringIO("b.inp  \tinput file\n"), out=io.StringIO())
    assert pr.cvalue("Input filespec") == "b.inp"


# ------------------------------------------------------------------ listing format
def test_long_polarity_error_lists_have_no_empty_record(tmp_path):
    # 33 conflicting pairs: every mechanism has exactly 33 bad P stations (3 full rows of 11)
    lines = ["t"] + [f"C{i:03d}   10.00   20.00C" for i in range(33)]
    lines += [f"D{i:03d}   10.00   20.00D" for i in range(33)]
    res = search(parse_inp(lines), FocmecParams(relweight=False, p_errors=33, trend=(0, 45, 45),
                                                plunge=(0, 45, 45), angle=(0, 45, 45)))
    res.write_lst(tmp_path / "x.lst")
    text = (tmp_path / "x.lst").read_text().splitlines()
    first = next(i for i, ln in enumerate(text) if "Polarity error at" in ln)
    total = next(i for i, ln in enumerate(text) if "Total P polarity weight" in ln and i > first)
    assert total - first == 6 and all(ln.strip() for ln in text[first:total])


# ------------------------------------------------------------------ geometry
def test_pt_axes_are_unit_vectors_with_the_right_plunge():
    p = geometry.trend_plunge_to_vector(np.radians(40.0), np.radians(30.0))
    t = np.cross(p, [0.0, 0.0, 1.0])
    t /= np.linalg.norm(t)
    a, n = (t + p) / math.sqrt(2), (t - p) / math.sqrt(2)
    pv, tv = geometry.pt_axes(a, n)
    assert np.linalg.norm(pv) == pytest.approx(1.0) and np.linalg.norm(tv) == pytest.approx(1.0)
    # focreps.f scales by sqrt(2), so this axis got |z| == 1 and was reported vertical
    assert np.degrees(geometry.vector_to_trend_plunge(pv)) == pytest.approx([40.0, 30.0])


def test_pt_axes_stay_single_precision():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    n = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    assert all(v.dtype == np.float32 for v in geometry.pt_axes(a, n))


def test_rake_is_finite_when_a_component_rounds_just_past_one():
    dsr = geometry.an_to_dsr(np.array([1.0000001, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    assert np.all(np.isfinite(dsr)) and np.degrees(dsr[2]) == pytest.approx(0.0)


def test_search_grid_follows_single_precision_rounding():
    # NINT of a REAL*4 ratio: btdel 8 at plunge 60 is an exact tie that float64 rounds the other way
    d = parse_inp(["t", "AAA    10.00   20.00C"])
    res = search(d, FocmecParams(p_errors=1, maxsol=0, trend=(0, 8, None), plunge=(60, 5, 60), angle=(0, 30, None)))
    assert res.nsol == 132  # the Fortran program's count for this run


# ------------------------------------------------------------------ dsretc
def test_moment_tensor_with_a_signed_zero_gives_a_valid_mechanism():
    # gfortran's SIGN honours -0.0, which made the original return NaN for this tensor
    rep, _ = ds.from_moment_tensor(0, 0, 0, 0, 0, 1)
    assert all(math.isfinite(v) for v in (*rep.dsr, *rep.aux, *rep.p, *rep.t))
    assert rep.dsr[0] == pytest.approx(90.0)


def test_dsretc_cli_reports_early_end_of_input(tmp_path, monkeypatch, capsys):
    from pyfocmec.cli.dsretc import main

    monkeypatch.chdir(tmp_path)
    assert main(["--interactive"], stdin=io.StringIO("comment\n"), stdout=io.StringIO()) == 1
    assert "end of input" in capsys.readouterr().err


# ------------------------------------------------------------------ number formatting
def test_fnum_prints_nan_and_infinity_like_gfortran():
    assert fnum(float("nan"), 8, 4) == "     NaN"
    assert fnum(float("inf"), 8, 4) == "Infinity"
    assert fnum(float("-inf"), 8, 4) == "    -Inf"
    assert fnum(float("-inf"), 9, 4) == "-Infinity"
    assert fnum(float("inf"), 2, 1) == "**"


# ------------------------------------------------------------------ synthetic input
def test_synth_comment_matches_synth_test_format():
    comment, _ = __import__("pyfocmec.synth", fromlist=["x"]).synthetic_inp(45.0, 30.0, 90.0, 30.0, 35.0)
    assert comment == "dip    45.00; strike     30.00; slip     90.00; vP/vS    1.7321"


# ------------------------------------------------------------------ command-line errors
def test_cli_reports_missing_input_file_in_one_line(tmp_path, monkeypatch, capsys):
    from pyfocmec.cli.focmec import main

    monkeypatch.chdir(tmp_path)
    assert main(["missing.inp"]) == 1
    assert capsys.readouterr().err == "focmec: error: cannot open missing.inp: no such file\n"


def test_cli_reports_bad_input_and_zero_increments(tmp_path, monkeypatch, capsys):
    from pyfocmec.cli.focmec import main

    monkeypatch.chdir(tmp_path)
    (tmp_path / "bad.inp").write_text("t\nAAA    1 2.5   4 0.0C\n")
    assert main(["bad.inp"]) == 1
    assert "cannot read azimuth" in capsys.readouterr().err
    (tmp_path / "ok.inp").write_text("t\nAAA    10.00   20.00C\n")
    assert main(["ok.inp", "--trend", "0", "0", "355"]) == 1
    assert "increment" in capsys.readouterr().err


def test_cli_errors_name_missing_optional_packages(capsys):
    from pyfocmec.cli.errors import clean_errors

    @clean_errors("prog")
    def needs_matplotlib():
        raise ModuleNotFoundError("No module named 'matplotlib'", name="matplotlib")

    assert needs_matplotlib() == 1
    assert "pip install 'pyfocmec[plot]'" in capsys.readouterr().err


def test_cli_debug_variable_keeps_the_traceback(monkeypatch):
    from pyfocmec.cli.focmec import main

    monkeypatch.setenv("PYFOCMEC_DEBUG", "1")
    with pytest.raises(FileNotFoundError):
        main(["definitely-missing.inp"])
