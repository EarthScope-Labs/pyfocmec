"""Polarities plus amplitude ratios: Sakhalin Island earthquake, 12 May 1990.

Shows the whole chain:  ratio-prep -> prep -> search -> plot.
The TauP step needs obspy (pip install pyfocmec[taup]); without it the example uses the ray table
that ``ratio_prep`` wrote for this event.  Run from this directory:  python run.py
"""

from pyfocmec import FocmecParams, prep, read_inp, search

vals, stations = prep.read_hypocenter_file("hyp_sak.txt")
locs = prep.read_station_locations("stations.loc")
try:
    table = prep.ratio_prep(*vals[:3], stations, locs, prep.TauP("nocrust"),
                            "Sakhalin: 12 May 1990  ak135 with no crust")
    print("computed take-off angles and travel times with obspy.taup")
except ImportError:
    # the table ratio_prep wrote for this event is shipped with the example
    table = prep.read_ray_table("focmec_prep_8sta.inp")
    print("obspy not installed; using the shipped ray table focmec_prep_8sta.inp")
comment, lines, _ = prep.focmec_prep(table, open("sakhalin_8sta.dat").read().splitlines())
open("my_focmec_8sta.inp", "w").write("\n".join([comment] + lines) + "\n")
inp = "my_focmec_8sta.inp"
print("built", inp)

data = read_inp(inp)
params = FocmecParams(
    total_errors=0.2,      # weighted polarity errors allowed
    ratio_error=0.16,      # max |log10 observed - calculated|
    n_ratio_errors=1,      # ratios allowed to exceed that
    maxsol=500,
)
result = search(data, params)
print(f"{result.nsol} acceptable solutions")
best = min(result.solutions, key=lambda s: s.ratio_rms)
print(f"lowest ratio rms {best.ratio_rms:.4f}: dip/strike/rake {best.dip:.1f}/{best.strike:.1f}/{best.rake:.1f}")
result.write_lst("focmec.lst")
result.write_out("focmec.out")

try:
    from pyfocmec.plot import focplt, ratplt

    ax = focplt(inp, result, dsr=(best.dip, best.strike, best.rake), nodal_surfaces=True)
    ax.figure.savefig("sakhalin.png", dpi=150)
    ratplt(inp, (best.dip, best.strike, best.rake)).figure.savefig("sakhalin_ratios.png", dpi=150)
    print("wrote sakhalin.png and sakhalin_ratios.png")
except ImportError:
    pass
