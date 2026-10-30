import pytest

pytest.importorskip("obspy")

from conftest import FIXTURES
from pyfocmec import prep

PREP = FIXTURES / "prep"


def test_taup_nocrust_matches_original_table():
    vals, stations = prep.read_hypocenter_file(FIXTURES / "prep" / "hyp_sak.txt")
    locs = prep.read_station_locations(FIXTURES / "inputs" / "stations.loc")
    table = prep.ratio_prep(*vals[:3], stations, locs, prep.TauP("nocrust"), "Sakhalin")
    ref = prep.read_ray_table(PREP / "focmec_prep_8sta.inp")
    assert table.vsvp_surface == pytest.approx(ref.vsvp_surface, abs=1e-3)
    assert table.vpvs_focus == pytest.approx(ref.vpvs_focus, abs=1e-3)
    for a, b in zip(table.rows, ref.rows):
        assert a.sta == b.sta
        for f in ("ptoa", "stoa", "pemrg", "semrg"):
            assert getattr(a, f) == pytest.approx(getattr(b, f), abs=0.25)
        assert a.tp == pytest.approx(b.tp, abs=0.2) and a.ts == pytest.approx(b.ts, abs=0.2)


def test_taup_builtin_model():
    t = prep.TauP("ak135")
    r = t.rays(60.0, 10.0)
    assert 0 < r["ptoa"] < 90 or r["ptoa"] > 90
    assert r["tp"] < r["ts"]
