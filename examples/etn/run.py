"""Polarity-only example: Chapman et al. (1997) event 12 (Eastern Tennessee, 1987-03-27).

Run from this directory:  python run.py
"""

from pyfocmec import FocmecParams, read_inp
from pyfocmec import search

data = read_inp("8703270729.inp")
print(f"{data.npol} polarities read ({data.count_pol(0)} P)")

# unweighted ("UW") search allowing 4 P-polarity errors on the default 5-degree grid
params = FocmecParams(relweight=False, p_errors=4, comment="ETN event 12, 4 UW errors")
result = search(data, params)
print(f"{result.nsol} acceptable solutions")
for sol in list(result.solutions)[:5]:
    print(f"  dip {sol.dip:6.2f}  strike {sol.strike:6.2f}  rake {sol.rake:7.2f}   P/T trend/plunge "
          f"{sol.p_trend:6.1f}/{sol.p_plunge:4.1f}  {sol.t_trend:6.1f}/{sol.t_plunge:4.1f}")

# the same files the Fortran program writes
result.write_lst("focmec.lst")
result.write_out("focmec.out")

try:
    from pyfocmec.plot import focplt

    ax = focplt("8703270729.inp", result, emergent=True, title="ETN event 12")
    ax.figure.savefig("etn.png", dpi=150)
    print("wrote etn.png")
except ImportError:
    print("install matplotlib (pip install pyfocmec[plot]) to draw the plot")
