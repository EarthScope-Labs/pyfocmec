# pyfocmec

Python port of **FOCMEC** (FOCal MEChanism determinations) by Arthur Snoke (Virginia Tech).  The
original Fortran code can be found at: https://github.com/iris-edu-legacy/focmec

FOCMEC finds the double-couple focal mechanisms that fit observed **P, SV and SH polarities** and
**SV/P, SH/P and SV/SH amplitude ratios**.  It searches the focal sphere systematically on a regular
grid and reports every mechanism whose polarity and amplitude-ratio misfits are within limits you set.
`pyfocmec` reproduces the original Fortran program (including its `focmec.lst` / `focmec.out` file
formats) and bundles the companion tools of the original package, with no compiler or Makefile needed.

```
pip install pyfocmec               # core: needs only numpy
pip install "pyfocmec[plot]"       # + matplotlib for focal-sphere plots
pip install "pyfocmec[taup]"       # + obspy for travel-time/take-off-angle calculation
```

Python 3.10 or newer.

## Quick start

### Command line

```
focmec 8703270729.inp --uw --p-errors 4          # writes focmec.lst and focmec.out
focmec data.inp --total-errors 0.2 --ratio-error 0.16 --n-ratio-errors 1 \
       --trend 0 5 355 --plunge 0 5 90 --angle 0 5 175 --maxsol 500
focmec data.inp --params run.toml                 # parameters from a TOML file
```

The original interactive dialogue is still there, so the driver scripts that came with FOCMEC work
with the matching `--interactive` (or `-i`) command in place of the Fortran binary: the `rfocmec_*`
scripts with `focmec --interactive`, the `rdsretc_*` scripts with `pyfocmec dsretc --interactive`:

```
focmec --interactive < answers.txt
```

The prompt dialogues of `focplt`, `ratplt`, `ratio_prep`, `focmec_prep` and `ttimes` are not replayed:
use the flags of the corresponding `pyfocmec` subcommands instead.  Errors (bad input, missing files,
a missing optional package) are reported in one line and exit with status 1; set `PYFOCMEC_DEBUG=1` to get
the traceback.

### Python

```python
from pyfocmec import read_inp, search, FocmecParams

data = read_inp("8703270729.inp")
result = search(data, FocmecParams(relweight=False, p_errors=4))
print(result.nsol, "solutions")
for s in result.solutions:
    print(s.dip, s.strike, s.rake, s.p_trend, s.p_plunge, s.bad_p)

result.write_lst("focmec.lst")      # same files as the Fortran program
result.write_out("focmec.out")
rows = result.solutions.to_records()  # list of dicts, e.g. pandas.DataFrame(rows)
```

`FocmecParams` mirrors the answers the program asks for:

| parameter | meaning | default |
|---|---|---|
| `relweight` | weight polarity errors by the theoretical radiation factor (RW); `False` counts each as 1 (UW) | `True` |
| `p_errors`, `sv_errors`, `sh_errors` | allowed polarity errors per phase | 0 |
| `total_errors` | allowed total polarity errors; a single limit when P and SV/SH polarities are both present. With one kind of polarity it stands in for any per-phase limit left at 0 | – |
| `ratio_error`, `n_ratio_errors` | largest acceptable \|log10 observed − calculated\| and how many ratios may exceed it | 0.6, 0 |
| `cutp`, `cuts` | P / S radiation-factor cutoffs: ratios closer to a nodal surface are flagged `NUM`, `DEN` or `N&D` | 0.1, 0.1 |
| `trend`, `plunge`, `angle` | `(min, increment, max)` in degrees of the search over B-axis trend, B-axis plunge and the "Angle" of the A axis in the plane perpendicular to B | `(0,5,355)`, `(0,5,90)`, `(0,5,175)` |
| `maxsol` | stop after this many acceptable solutions (`0`: no limit) | 100 |
| `emergent` | use emergent polarity picks (`+ - l r f b`); `None` means include them, like answering the original prompt with its default | `None` |

## Input format

Line 1 is a comment.  Then one pick per line:

```
GBTN     9.00  148.00C                  iP        polarity: station, azimuth, take-off angle, key; comment from col 40
BLA     44.80   31.00H  0.8847 R  28.06 1.8225 …  ratio:    station, azimuth, take-off, key, log10 ratio,
                                                            numerator polarity, denominator take-off, Vp/Vs at source
```

Polarity keys — P: `C`/`U` up, `D` down; SV: `F` forward, `B` backward; SH: `L`/`<` left, `R`/`>` right
(left and right as seen from the station looking toward the event, i.e. with your back to the source);
emergent picks: `+ - l r f b` (`+`/`-` P, `l`/`r` SH, `f`/`b` SV).  Ratio keys — `V` SV/P, `H` SH/P, `S` SV/SH.  Azimuth is from the source to the
station, take-off angles are from straight down.  See the original manual for details.

## The tools

Everything in the original package that is still useful is here.  Subcommands of `pyfocmec`:

| original program | here | notes |
|---|---|---|
| `focmec` | `focmec` / `pyfocmec search`, `pyfocmec.search()` | vectorised with numpy; identical solutions and output text |
| `dsretc` | `pyfocmec dsretc`, `pyfocmec.dsretc` | dip/strike/rake ↔ P/T axes ↔ moment tensor, **and the B trend/plunge/Angle needed to run FOCMEC on one fixed mechanism** |
| `radiat` | `pyfocmec radiat`, `radiation_pattern()` | P/SV/SH radiation factors |
| `freesurf`, `grndmtn` | `pyfocmec freesurf`, `free_surface()` | free-surface amplitude corrections |
| `synth_test` | `pyfocmec synth`, `synthetic_inp()` | synthetic input for a known mechanism (handy for testing) |
| `ratio_prep`, `ratio_prep_1layer` | `pyfocmec ratio-prep`, `prep.ratio_prep()` | take-off/emergence angles and travel times; half-space, or any 1-D model through `obspy.taup` (`[taup]`): `iasp91`, `ak135`, `nocrust` (ak135 without the crust, as in the original distribution) |
| `focmec_prep` | `pyfocmec prep`, `prep.focmec_prep()` | picks + amplitudes → FOCMEC input, with free-surface and optional Q corrections |
| `hypo2foc` | `pyfocmec hypo2foc`, `prep.hypo2foc()` | hypoellipse archive → FOCMEC input files |
| `convert_input` | `pyfocmec convert` | pre-2017 input → current format |
| `focplt`, `ratplt` | `pyfocmec focplt` / `ratplt`, `pyfocmec.plot` | matplotlib, following the original's symbols, sizes and line styles: polarities (with `E` error boxes), any number of solution sets from `focmec.out` files or `--dsr` mechanisms, each with its own dash pattern and line width, as P nodal planes or SH/SV nodal surfaces, P/T/B axes; ratio misfit. Output to PDF/PNG/SVG |

Not carried over: the SAC-graphics-format (SGF) layer (`sgftops`, `sgftopdf`, `sgftoeps`, `sgftox`, `plsgf`,
`symbol`, `symdemo`, `vwbeetle`), replaced by matplotlib; `ttimes` and the `iaspei-tau` tables, replaced by
`obspy.taup`; `epicen`, which never compiled in the original distribution; the `.lst` listings written by
`ratio_prep`, `focmec_prep` and `synth_test` (use `--log`, which has a different layout); the station-elevation
table of `ratio_prep_1layer`; the `ratplt` listing of observed, calculated and difference values (available
as `Solution.ratios`).

A typical workflow with amplitude ratios:

```
pyfocmec ratio-prep hypocenter.txt --stations stations.loc --model nocrust -c "My event" -o table.txt
pyfocmec prep table.txt picks.dat -o focmec.inp
focmec focmec.inp --total-errors 0.2 --ratio-error 0.16 --n-ratio-errors 1
pyfocmec focplt --inp focmec.inp --out focmec.out -o solutions.pdf
```

`examples/` has runnable versions: `examples/etn` (polarities only) and `examples/sakhalin` (polarities plus
ratios through the whole chain).

## How closely does it match the Fortran program?

`tests/` replays 18 runs (the sample-run driver scripts of the original distribution plus a few edge cases)
through `focmec --interactive` and compares with output of the original program (built from the 2017
release).  The set of solutions is always identical, and every line of the `.lst` and `.out` files is
byte-identical except one line where a log ratio differs in its last printed digit (the original's
single-precision trigonometry rounds differently from numpy's) and the deliberate P/T-axis difference
below.  In `--interactive` mode (or with `--float32` / `dtype=np.float32`) `pyfocmec` computes in
single precision like the original; by default it uses double precision, which can accept or reject a
solution that lies exactly on a nodal plane differently.  `dsretc` is reproduced byte for byte (50 test
cases including moment tensors, multi-run sessions and the ASCII beachball).

### Differences you should know about

Where the original clearly contains a bug and the result is a defensible improvement, `pyfocmec` corrects
it.  Each of these is also listed in `CHANGELOG.md`.

* **P/T axes in the `focmec` listing.**  `focreps.f` builds the P and T vectors with `sqrt(2)` instead of
  `1/sqrt(2)` (`dsrin.f`, used by `dsretc`, has it right).  The vectors then have length 2, and when an axis
  has a true plunge of 30° its vertical component is exactly 1 and the axis is printed as vertical (plunge
  90°).  `pyfocmec` prints the correct plunge.
* **`focmec_prep` 80-character buffer overflow.**  For ratio lines whose amplitude text is long (such as the
  Sakhalin example with all of `num,den,Qs,Qp,fs,fp`) the original overflows an 80-character buffer: the
  first station gets a wrong azimuth and the P-wave Q term is dropped.  The published Sakhalin input file
  `focmec_8sta.inp` contains these artifacts.  `pyfocmec prep` computes the corrected values, and agrees
  with the original byte for byte whenever the original does not overflow.  Amplitude text longer than the
  200-character working buffer is an error rather than silently truncated.
* **Q correction of SV/SH ratios.**  `focmec_prep` uses the P travel time in the second attenuation term
  even when the denominator is an S wave.  `pyfocmec prep` uses the S travel time for `S` ratios.
* **Emergent SV picks.**  The original converts emergent `f` and `b` picks but never lets them reach that
  code, so they were silently dropped.  `pyfocmec` includes them (and asks the emergent question when they
  are the only emergent picks).
* **Numbers without a decimal point.**  In a fixed-column field such as the azimuth, Fortran reads `180`
  as 1.80.  `pyfocmec` reads 180 and warns.  Blanks inside a number are an input error, where Fortran
  silently drops them.  A ratio line with no ratio data is an input error, where Fortran reuses the values
  of the previous line.
* **`dsretc`.**  A moment tensor with a zero `MTP` or `MRP` component can make the original's eigen-solver
  return NaN or swap the two nodal planes (gfortran's `SIGN` honours `-0.0`); `pyfocmec` gives a valid
  decomposition.  The `Angle` is printed where the original prints `NaN`.
* **`synth_test`** wrote the text `sv/p` where the current input format has the Vp/Vs field;
  `pyfocmec synth` writes a valid field.
* **Listing format.**  `pyfocmec` does not reproduce the empty records the original writes after a list of
  polarity errors whose length is a multiple of 11 (beyond the first 22).  The "Calc." polarity letter of a
  ratio whose numerator radiation factor is exactly zero repeats the previous ratio's letter within a
  solution, as in the original, but starts blank in each solution instead of inheriting the last letter of
  the previous one.
* **Rake from `an2dsr`.**  One branch of `an2dsr.f` takes the unclamped `acos` of a value that can round to
  just above 1, giving a NaN rake; `pyfocmec` clamps it.
* **Epicentral distance and azimuth** stay finite for an event at a pole or an antipodal station, where the
  original divides by zero.
* **Station names are matched exactly** (still ignoring case), not by prefix, in `ratio-prep`.
* **Travel times come from `obspy.taup`**, not the `iaspei-tau` tables, and the default model is `ak135`
  where the original used IASP91.
* **Plots** are drawn with matplotlib rather than through SGF/`sgftopdf`.  They follow the original closely:
  the same symbol set, sizes (in the inches of the original, on its 2.3 in sphere), dash patterns and label
  positions.  The shapes and the lettering are redrawn, and line weights and the phase of dashes can differ by
  a fraction of a point.  `focplt` also offers a shaded compressional quadrants
  option (`--beachball`) and a hemisphere caption (`--caption`), and `ratplt` a symbol legend (`--legend`).

## Citation and credit

FOCMEC was written by Arthur Snoke.  If you use it, please cite

* Snoke, J. A., J. W. Munsey, A. G. Teague and G. A. Bollinger (1984), A program for focal mechanism
  determination by combined use of polarity and SV-P amplitude ratio data, *Earthquake Notes* 55(3), 15.
* Snoke, J. A. (2003), FOCMEC: FOCal MEChanism determinations, in *International Handbook of Earthquake and
  Engineering Seismology* (W. H. K. Lee, H. Kanamori, P. C. Jennings and C. Kisslinger, eds.), Academic Press,
  San Diego, Chapter 85.12.

The numerical code is a port of the 2017 release of the Fortran package; consult the original manual
(`focmec_manual.pdf` in that [distribution](https://github.com/iris-edu-legacy/focmec)) for the method,
conventions and examples.

## Development

```
python -m venv .venv && .venv/bin/pip install -e ".[dev,plot,taup]"
.venv/bin/pytest
```

## License

MIT licensed (see `LICENSE`).
