import subprocess
import sys

import numpy as np
import pytest

import pyfocmec
from pyfocmec import FocmecParams, geometry, parse_inp, read_inp, search
from pyfocmec.radiation import free_surface, free_surface_corrections, radiation_pattern
from pyfocmec.synth import synthetic_inp

from conftest import FIXTURES

INPUTS = FIXTURES / "inputs"


def test_fault_vectors_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(200):
        t, p, a = rng.uniform(0, 2 * np.pi), rng.uniform(0, np.pi / 2), rng.uniform(0, np.pi)
        A, N = geometry.fault_vectors(np.array([p]), np.array([t]), np.array([a]))
        assert np.allclose(np.linalg.norm(A), 1) and np.allclose(np.linalg.norm(N), 1)
        assert abs(float(A[0] @ N[0])) < 1e-12
        dsr = geometry.an_to_dsr(A[0], N[0])
        A2, N2 = geometry.dsr_to_an(*dsr)
        # same fault up to the A/N interchange ambiguity of the double couple
        same = np.allclose(A2, A[0], atol=1e-6) and np.allclose(N2, N[0], atol=1e-6)
        swapped = np.allclose(A2, N[0], atol=1e-6) and np.allclose(N2, A[0], atol=1e-6)
        assert same or swapped or np.allclose(A2, -A[0], atol=1e-6) or np.allclose(A2, -N[0], atol=1e-6)


def test_trend_plunge_roundtrip():
    v = geometry.trend_plunge_to_vector(np.radians(123.0), np.radians(34.0))
    tp = np.degrees(geometry.vector_to_trend_plunge(v))
    assert tp == pytest.approx([123.0, 34.0])


def test_parse_inp_counts():
    d = read_inp(INPUTS / "focmec_8sta.inp")
    assert (d.count_pol(0), d.count_pol(1), d.count_pol(2), d.nrat) == (8, 7, 8, 11)
    assert d.vpvs == pytest.approx(1.8225)
    d = read_inp(INPUTS / "8703270729.inp")
    assert d.has_emergent and d.npol == 37
    d2 = parse_inp(str(INPUTS / "8703270729.inp"), emergent=False)
    assert d2.npol < d.npol


def test_api_matches_documented_solution_counts():
    # ETN event, unweighted, 4 P errors allowed: 13 solutions (see the FOCMEC manual)
    d = read_inp(INPUTS / "8703270729.inp")
    res = search(d, FocmecParams(relweight=False, p_errors=4, emergent=True))
    assert res.nsol == 13
    s = res.solutions[0]
    assert (round(s.dip, 2), round(s.strike, 2), round(s.rake, 2)) == (90.00, 274.91, 50.00)
    assert len(res.solutions.to_records()) == 13


def test_maxsol_truncates():
    d = read_inp(INPUTS / "focmec_8sta-noratios.inp")
    res = search(d, FocmecParams(maxsol=5, total_errors=0.2))
    assert res.nsol == 5 and res.truncated


def test_float32_and_float64_agree_on_solution_set():
    d = read_inp(INPUTS / "focmec_8sta.inp")
    prm = FocmecParams(total_errors=0.2, ratio_error=0.16, n_ratio_errors=1, maxsol=500)
    a = search(d, prm)
    b = search(d, prm, dtype=np.float32)
    assert a.nsol == b.nsol


@pytest.mark.parametrize("dsr", [(50, 90, 0), (30, 200, 80), (75, 10, -120)])
def test_synthetic_roundtrip_recovers_mechanism(dsr, tmp_path):
    comment, lines = synthetic_inp(*dsr, 35.0, 40.0, vpvs=1.73, n_azimuths=24)
    f = tmp_path / "s.inp"
    f.write_text(comment + "\n" + "\n".join(lines) + "\n")
    res = search(read_inp(f), FocmecParams(trend=(0, 5, None), plunge=(0, 5, 90), angle=(0, 5, None),
                                           maxsol=100000, ratio_error=0.4, n_ratio_errors=6,
                                           cutp=0.05, cuts=0.05, total_errors=1.0))
    assert res.nsol > 0
    # the true mechanism (or its auxiliary plane) is within one grid step of some solution
    true = np.array(dsr, float)
    best = min(
        min(np.abs(np.array([s.dip, s.strike, s.rake]) - true).sum(),
            np.abs(np.array([s.aux_dip, s.aux_strike, s.aux_rake]) - true).sum())
        for s in res.solutions
    )
    assert best < 25.0


def test_radiation_pattern_matches_fortran():
    pr, svr, shr = radiation_pattern(55, 120, -40, 33, 35, 40)
    assert (round(float(pr), 3), round(float(svr), 3), round(float(shr), 2)) == (-0.457, -0.520, 0.75)


def test_free_surface_matches_fortran():
    res = free_surface_corrections(0.5572, 20, 22)
    assert res["P_Z"] == pytest.approx(1.86468673, abs=1e-6)
    assert res["P_R"] == pytest.approx(0.752342403, abs=1e-6)
    assert res["SV_R/P_Z"] == pytest.approx(1.02343714, abs=1e-6)
    sup = free_surface_corrections(0.5572, 10, 40)  # post-critical SV
    assert sup["supercritical"] and sup["S_Z"] == pytest.approx(1.55144525, abs=1e-5)
    assert free_surface(1, 0.5, 0.0)[:2] == (0.0, 2.0)


def test_cli_noninteractive_matches_fortran_table(workdir):
    from numcmp import lines_match

    inp = str(workdir / "focmec_8sta-noratios.inp")
    out = workdir / "cli.out"
    p = subprocess.run([sys.executable, "-m", "pyfocmec.cli.focmec", inp, "--uw", "--total-errors", "0",
                        "--maxsol", "1000", "-o", str(out), "--lst", str(workdir / "cli.lst"), "-q", "--float32"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    got = out.read_text().splitlines()
    ref = (FIXTURES / "ref" / "focmec_8sta-noratios.out").read_text().splitlines()
    # headers differ (comment, file name); the solution table must agree
    g, r = got[next(i for i, x in enumerate(got) if "Dip" in x):], ref[next(i for i, x in enumerate(ref) if "Dip" in x):]
    assert len(g) == len(r) > 10
    assert all(lines_match(a, b) for a, b in zip(g, r, strict=True))


def test_params_toml(tmp_path, workdir):
    (workdir / "run.toml").write_text('relweight = false\np_errors = 4\ncomment = "from toml"\n')
    p = subprocess.run([sys.executable, "-m", "pyfocmec.cli.focmec", str(workdir / "8703270729.inp"),
                        "--params", str(workdir / "run.toml"), "-q", "--lst", str(workdir / "a.lst"),
                        "-o", str(workdir / "a.out")], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    text = (workdir / "a.lst").read_text()
    assert "from toml" in text and "There are  13 acceptable solutions" in text


def test_version():
    assert pyfocmec.__version__
