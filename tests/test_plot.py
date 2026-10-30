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


def _record_symbols(monkeypatch):
    calls = []
    real = plot.draw_symbol

    def spy(ax, x, y, ns, height, *a, **kw):
        calls.append((ns, height))
        return real(ax, x, y, ns, height, *a, **kw)

    monkeypatch.setattr(plot, "draw_symbol", spy)
    return calls


def test_symbol_shapes():
    from pyfocmec import _symbols as sym

    pts = {k: {p for stroke in sym.SHAPES[k] for p in stroke} for k in (1, 2, 5, 12)}
    assert len(pts[1] - {(0, 0), (0, 10)}) == 8  # octagon (P compression; the tick reaches the top edge)
    assert len(pts[2] - {(0, 0)}) == 3  # triangle (P dilatation)
    assert pts[12] == {(-10, 10), (10, 10), (-10, -10), (10, -10)}  # hourglass
    # a symbol of height 21 spans +-10; turning the arrow 90 degrees counter-clockwise moves its tip west
    tip = sym.strokes(6, 21.0, angle=90.0)[0][1]
    assert tuple(np.round(tip, 9)) == (-10.0, 0.0)


def test_ratplt_symbols_grow_linearly_with_scale_and_misfit(monkeypatch):
    # ratplt.f draws a symbol of height scale*10**|d| (and a reference one of scale*10**error)
    inp = str(INPUTS / "focmec_8sta.inp")
    dsr = (76.43, 59.08, -64.23)
    runs = {}
    for scale in (0.2, 0.4):
        calls = _record_symbols(monkeypatch)
        plot.ratplt(inp, dsr, error=0.5, scale=scale)
        runs[scale] = list(calls)[1:]  # the first symbol is the plus at the centre of the sphere
    small, big = runs[0.2], runs[0.4]
    assert [ns for ns, _ in small] == [ns for ns, _ in big]
    assert {ns for ns, _ in small} <= {0, 1, 5, 12}
    for (ns, h1), (_, h2) in zip(small, big):
        assert h2 == pytest.approx(2 * h1)
    ref = [h for ns, h in small if ns == 12]
    assert ref and all(h == pytest.approx(0.2 * 10**0.5 / plot._R0) for h in ref)
    d = np.log10(np.array([h for ns, h in small if ns != 12]) * plot._R0 / 0.2)  # |o-c| of every ratio
    assert np.all(d >= 0) and np.all(d < 1.0)


def test_polarity_symbols_and_error_box(tmp_path, monkeypatch):
    inp = tmp_path / "t.inp"
    inp.write_text(
        "test\n"
        + "".join(f"{s:<4}{az:8.2f}{toa:8.2f}{c}\n" for s, az, toa, c in (
            ("A", 10.0, 40.0, "C"), ("A", 10.0, 40.0, "E"), ("B", 120.0, 50.0, "D"), ("C", 200.0, 60.0, "F"),
            ("D", 300.0, 70.0, "<"), ("E", 100.0, 30.0, "+"))))
    ax = plot.focal_sphere()
    calls = _record_symbols(monkeypatch)
    plot.plot_polarities(inp, ax=ax, size=0.2, errors=True, emergent=True, emergent_size=0.1)
    u = 1.0 / plot._R0
    assert [ns for ns, _ in calls] == [1, 0, 2, 6, 6, 3]  # E box follows its pick
    assert [h for _, h in calls] == pytest.approx([0.2 * u, 0.2 * u, 0.2 * u, 0.2 * u, 0.2 * u, 0.1 * u])
    calls.clear()
    plot.plot_polarities(inp, ax=ax, phases=("P",))  # no error boxes, no emergent picks, no SV/SH
    assert [ns for ns, _ in calls] == [1, 2]


def test_per_set_styles(tmp_path):
    dsr = (76.43, 59.08, -64.23)
    ax = plot.focplt(sets=[dict(dsr=dsr, surface="SH", dash=(0.15, 0.05), lw=2)])
    assert not [ln for ln in ax.lines if len(ln.get_xdata()) == 181]  # no fault planes, only the SH surface
    assert len(ax.collections) == 1
    ax = plot.focplt(sets=[dict(dsr=dsr), dict(dsr=dsr, dash=(0.15, 0.02), lw=2)])
    planes = [ln for ln in ax.lines if len(ln.get_xdata()) == 181]
    assert len(planes) == 4
    assert [ln.get_linewidth() for ln in planes] == pytest.approx([plot._pen(1)] * 2 + [plot._pen(2)] * 2)
    assert planes[0].is_dashed() is False and planes[2].is_dashed() is True


def test_focplt_cli_sets_and_ratplt_label(tmp_path):
    from pyfocmec.cli.plot import focplt_main, ratplt_main

    out = tmp_path / "f.png"
    assert focplt_main(["--inp", str(INPUTS / "focmec_8sta.inp"), "--dsr", "76.43", "59.08", "-64.23",
                        "--dsr", "77.05", "54.08", "-59.13", "--surface", "SH", "--phases", "SH", "--label", "c)",
                        "-o", str(out)]) == 0
    assert out.stat().st_size > 2000
    out2 = tmp_path / "r.png"
    assert ratplt_main([str(INPUTS / "focmec_8sta.inp"), "--dsr", "76.43", "59.08", "-64.23", "--error", "0.16",
                        "--scale", "0.3", "--label", "a)", "-o", str(out2)]) == 0
    assert out2.stat().st_size > 2000
