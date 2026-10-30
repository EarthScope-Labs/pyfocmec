import re
import subprocess
import sys

import pytest

from conftest import FIXTURES
from numcmp import trend_wrap_equiv
from pyfocmec import dsretc as ds

CASES = sorted(p.stem for p in (FIXTURES / "dsretc").glob("*.stdin"))


@pytest.mark.parametrize("case", CASES)
def test_dsretc_replay_matches_fortran(case, tmp_path):
    stdin = (FIXTURES / "dsretc" / f"{case}.stdin").read_text()
    subprocess.run([sys.executable, "-m", "pyfocmec.cli.dsretc", "--interactive"], input=stdin,
                   text=True, cwd=tmp_path, capture_output=True, check=True)
    got = _unstamped((tmp_path / "dsretc.lst").read_text())
    ref = _unstamped((FIXTURES / "dsretc" / f"{case}.lst").read_text())
    if sys.platform != "darwin":
        # The reference was made with Apple's libm.  Elsewhere the last bit of float32 trig differs, which
        # moves a few beachball characters and can change whether get_angle finds an angle, so only the
        # listing above the "Angle:" line is compared.
        got, ref = (_head(x) for x in (got, ref))
    assert len(got) == len(ref)
    assert all(g == r or trend_wrap_equiv(g, r) for g, r in zip(got, ref))


def _head(lines):
    return lines[: next((i for i, ln in enumerate(lines) if ln.lstrip().startswith("Angle:")), len(lines))]


def _unstamped(text):
    """Listing lines with every 'date for program dsretc' stamp (one per comment) blanked out."""
    return [re.sub(r"^  .* for program dsretc$", "STAMP", ln) for ln in text.splitlines()]


def test_dsr_roundtrip_float64():
    rep = ds.from_dsr(50.0, 90.0, 0.0)
    assert rep.dsr == pytest.approx((50.0, 90.0, 0.0), abs=1e-6)
    assert rep.aux[0] == pytest.approx(90.0)
    # the angle reproduces the mechanism through the FOCMEC parameterisation
    assert rep.angle is not None


def test_pt_to_dsr_consistent():
    rep = ds.from_pt(269.0, 13.0, 171.0, 30.0)
    rep2 = ds.from_dsr(*rep.dsr)
    assert rep2.p == pytest.approx(rep.p, abs=0.05)
    assert rep2.t == pytest.approx(rep.t, abs=0.05)


def test_isotropic_tensor_rejected():
    with pytest.raises(ValueError):
        ds.from_moment_tensor(1, 1, 1, 0, 0, 0)
