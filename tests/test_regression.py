"""Replay the original FOCMEC driver scripts and compare with output of the Fortran program."""

import json
import subprocess
import sys

import pytest

from numcmp import compare_files

from conftest import FIXTURES

MANIFEST = json.load(open(FIXTURES / "ref" / "manifest.json"))

# Lines (0-based, after the time stamp) that may differ from the Fortran output in the last printed
# digit: numpy and libm round float32 trig differently.  Everything else must be byte-identical,
# apart from the documented P/T axis divergence handled in numcmp.pt_axis_divergence.
NOISE = {
    "rfocmec_8sta": {"lst": {90}},
}


@pytest.mark.parametrize("script", sorted(MANIFEST))
def test_interactive_replay_matches_fortran(script, workdir):
    stdin = (FIXTURES / "ref" / f"{script}.stdin").read_text()
    p = subprocess.run([sys.executable, "-m", "pyfocmec.cli.focmec", "--interactive", "-q"],
                       input=stdin, text=True, capture_output=True, cwd=workdir)
    assert p.returncode == 0, p.stderr
    names = MANIFEST[script]
    noise = NOISE.get(script, {})
    compare_files(workdir / "focmec.lst", FIXTURES / "ref" / names["lst"], noise=noise.get("lst", ()))
    compare_files(workdir / names["out"], FIXTURES / "ref" / names["out"], noise=noise.get("out", ()))
