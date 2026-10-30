import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

from conftest import FIXTURES
from pyfocmec import FocmecParams, plot, read_inp, search

INPUTS = FIXTURES / "inputs"


def test_projection_equal_area():
    # centre, and horizon at the primitive circle
    x, y = plot.project(0.0, 0.0)
    assert (float(x), float(y)) == (0.0, 0.0)
    x, y = plot.project(90.0, 90.0)
    assert float(x) == pytest.approx(1.0) and float(y) == pytest.approx(0.0, abs=1e-12)
    # upgoing rays fold to the antipode
    x1, y1 = plot.project(0.0, 120.0)
    x2, y2 = plot.project(180.0, 60.0)
    assert (float(x1), float(y1)) == pytest.approx((float(x2), float(y2)))


def test_nodal_plane_on_unit_circle_ends():
    x, y = plot.nodal_plane_xy(90.0, 0.0)  # vertical N-S plane: a straight line through the centre
    assert np.allclose(x, 0.0, atol=1e-9)
    x, y = plot.nodal_plane_xy(45.0, 90.0)
    assert np.hypot(x[0], y[0]) == pytest.approx(1.0, abs=1e-6)


def test_focplt_and_ratplt_render(tmp_path):
    res = search(read_inp(INPUTS / "focmec_8sta.inp"),
                 FocmecParams(total_errors=0.2, ratio_error=0.16, n_ratio_errors=1))
    ax = plot.focplt(str(INPUTS / "focmec_8sta.inp"), res, dsr=(76.43, 59.08, -64.23), nodal_surfaces=True,
                     beachball=True)
    ax.figure.savefig(tmp_path / "a.png")
    assert (tmp_path / "a.png").stat().st_size > 5000
    ax2 = plot.ratplt(str(INPUTS / "focmec_8sta.inp"), (76.43, 59.08, -64.23))
    ax2.figure.savefig(tmp_path / "b.png")
    assert (tmp_path / "b.png").stat().st_size > 2000


def test_read_out_rows():
    rows = plot.read_out(FIXTURES / "ref" / "focmec_8sta.out")
    assert rows and rows[0]["dip"] == pytest.approx(76.43)
    assert rows[0]["ratio_total"] == 11


def test_cli_plot(tmp_path):
    from pyfocmec.cli.plot import focplt_main

    out = tmp_path / "f.png"
    assert focplt_main(["--inp", str(INPUTS / "8703270729.inp"), "--out", str(FIXTURES / "ref" / "focmec_UW-4.out"),
                        "--emergent", "-o", str(out)]) == 0
    assert out.exists()


def test_read_out_follows_solplt_header_rule(tmp_path):
    f = tmp_path / "x.out"
    # numbers in the header must not be mistaken for solutions; three columns are enough
    f.write_text(
        " 1 2 3 4 5 6 7\n"
        "    Dip   Strike   Rake\n"
        "   76.43   59.08  -64.23\n"
        "\n"
        "   10.00   20.00   30.00\n"
    )
    rows = plot.read_out(f)
    assert [(r["dip"], r["strike"], r["rake"]) for r in rows] == [(76.43, 59.08, -64.23), (10.0, 20.0, 30.0)]
    f.write_text("1 2 3\n")
    with pytest.raises(ValueError, match="Dip"):
        plot.read_out(f)


def test_ratplt_symbols_grow_linearly_with_the_misfit():
    # ratplt.f draws a symbol of height scale*10**|d|; scatter(s=) is an area, so s ~ (10**|d|)**2
    inp = str(INPUTS / "focmec_8sta.inp")
    ax = plot.ratplt(inp, (76.43, 59.08, -64.23), error=0.5, scale=0.2)
    refs = [c for c in ax.collections if tuple(c.get_edgecolor()[0][:3]) == (0.5, 0.5, 0.5)]
    assert refs and refs[0].get_sizes()[0] == pytest.approx(960 * 0.2 * (10**0.5) ** 2)
    symbols = [c for c in ax.collections if c not in refs]
    sizes = np.array([c.get_sizes()[0] for c in symbols])
    d = 0.5 * np.log10(sizes / (960 * 0.2))  # recovers |o-c| for every ratio
    assert np.all(d >= 0) and np.all(d < 1.0)  # misfits of a good mechanism are well below a decade
