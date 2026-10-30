# Changelog

pyfocmec is a port of the Fortran FOCMEC package by Arthur Snoke (2017 release). This file records only
where pyfocmec deliberately behaves differently from that reference. Fixes that bring the port back into
agreement with the Fortran are not listed.

## Unreleased

### Corrections of bugs in the original

- **P/T axes in the `focmec` listing.** `focreps.f` scales `A -/+ N` by `sqrt(2)` instead of `1/sqrt(2)`
  (`dsrin.f` has it right), so the P and T vectors have length 2. An axis with a true plunge of 30 degrees
  then has a vertical component of exactly 1 and was printed as vertical (plunge 90). pyfocmec prints
  the correct plunge. In the reference runs this changes five `Lower Hem. Trend, Plunge of P,T` lines in four of
  the listings (for example `180.00 90.00` becomes `40.56 30.00`).
- **Q correction of SV/SH ratios in `focmec_prep`.** The original uses the P travel time in the second
  attenuation term even when the denominator is an S wave, so equal Q and frequency still gave a
  correction. pyfocmec uses the S travel time for `S` ratios; `V` and `H` ratios are unchanged.
- **Emergent SV picks.** `focinp.f` converts emergent `f` and `b` picks, but the test in front of that code
  admits only `+ - l r`, so they were silently dropped. pyfocmec includes them and asks the
  emergent question when they are the only emergent picks.
- **NaN rake.** One branch of `an2dsr.f` takes the unclamped `acos` of a value that can round to just above 1.
  pyfocmec clamps it.
- **`focmec_prep` amplitude text** longer than the 200-character working buffer is an error, not silently
  truncated.

### Behaviour changes

- **Numbers without a decimal point** in a fixed-column field are read as written (`180` is 180) with a
  warning, where Fortran applies an implied decimal (1.80). Blanks inside a number are an input error
  naming the line and field, where Fortran silently drops them. A ratio line with no ratio data is an
  input error, where Fortran reuses the values of the previous line.
- **Listing format.** The empty records the original writes after a list of polarity errors whose length is a
  multiple of 11 (beyond the first 22) are not written. The "Calc." polarity letter of a ratio whose
  numerator radiation factor is exactly zero no longer carries over from the previous solution.
- **Epicentral distance and azimuth** stay finite for an event at a pole or an antipodal station, where the
  original divides by zero.

### Added

- Command-line errors (bad input, missing file, missing optional package) are reported in one line with exit
  status 1; `PYFOCMEC_DEBUG=1` gives the traceback.

## 0.1.0

### Baseline: faithful port of FOCMEC 2017

The numerical code, input parsing and output formats follow the Fortran sources (`src/*.f`, `fmsub/*.f`).
`focmec.lst`, `focmec.out` and `dsretc.lst` match the original program's output, apart from last-digit
single-precision noise. The original prompt dialogues can be replayed with `--interactive`, so the
distributed `r*` driver scripts still work. Except where listed below, behaviour, including the original's
quirks, is the same as the reference.

### Corrections of bugs in the original

- **`focmec_prep` 80-character buffer overflow.** With long amplitude text (e.g. `num,den,Qs,Qp,fs,fp`),
  the original truncated the line: the first station got a wrong azimuth and the P-wave Q term was
  dropped. pyfocmec computes the intended values. The published Sakhalin `focmec_8sta.inp` contains the
  original's artifacts, so it differs from pyfocmec's output.
- **`focmec_prep` polarity lines with trailing text.** The original wrote leftover buffer contents into the
  comment field. pyfocmec writes the pick's text starting at column 40.
- **`synth_test` Vp/Vs field.** The original wrote the text `sv/p` where the current input format expects
  the Vp/Vs value. `pyfocmec synth` writes a valid number.
- **`convert_input` output filename.** The original truncated the output name to the length of the input
  name. pyfocmec does not.
- **`dsretc` moment-tensor signed zero.** gfortran's `SIGN` follows the sign bit of `-0.0` in the EISPACK
  eigen-solver. As a result, a tensor with zero `MTP`/`MRP` components could give all-`NaN` output, or swap
  which plane is primary and which auxiliary. pyfocmec treats `-0.0` as positive and gives a valid
  decomposition.
- **`dsretc` Angle.** The `acos` argument is clamped to [-1, 1], so pyfocmec prints an Angle where the
  original printed `NaN`.

### Behaviour changes

- **Precision.** The Python API and the non-interactive CLI compute in double precision. `--interactive`,
  `--float32` and `dtype=np.float32` use single precision, like the original's REAL*4. Near nodal planes,
  the double-precision default can accept or reject a borderline solution differently.
- **Epicentral distance and azimuth** (`distaz`) use double precision throughout.
- **Station matching** in `ratio-prep` is exact by name (still case-insensitive). The original compared
  only the first characters of each record against the requested name, so a short name matched any longer
  one starting with it.
- **Travel times and take-off angles** come from `obspy.taup` (`[taup]` extra) instead of the iaspei-tau
  tables. Available models are `iasp91`, `ak135` and `nocrust` (ak135 without the crust, as in the original
  distribution). When the hypocentre file gives no vp/vs, the default model is `ak135`; the original used
  IASP91.
- **Prep-tool listings.** `ratio_prep`, `focmec_prep` and `synth_test` wrote `.lst` files. In pyfocmec this
  information goes to an optional `--log` file in a different format.
- **Plotting** uses matplotlib instead of the SAC-graphics (SGF) layer, writing PDF, PNG or SVG. The plots
  follow the original closely (same symbol set, sizes, dash patterns and label positions); shapes and lettering
  are redrawn and line weights and dash phase can differ slightly.

### Added

- Command-line flags and TOML parameter files (`--params`) as alternatives to the interactive dialogue.
- A Python API: `read_inp`, `search`, `FocmecParams`, `FocmecResult`, `Solution` (with `to_records()`),
  plus functions for every tool.
- A single `pyfocmec` command with subcommands: `search`, `dsretc`, `radiat`, `freesurf`, `synth`,
  `convert`, `ratio-prep`, `prep`, `hypo2foc`, `focplt`, `ratplt`.
- `dsretc` can be run non-interactively (`--dsr`, `--pt`, `--mt`).
- `focplt --beachball` and `--caption` and `ratplt --legend`, which have no counterpart in the original.

### Not ported

- The SGF graphics programs: `sgftops`, `sgftopdf`, `sgftoeps`, `sgftox`, `plsgf`, `symbol`, `symdemo`,
  `vwbeetle`.
- `ttimes` and the iaspei-tau tables (replaced by `obspy.taup`).
- `epicen`, which did not compile in the original distribution.
- The `ratio_prep_1layer` station-elevation correction table and the `ratplt` listing file.
- Interactive prompt replay for `focplt`, `ratplt`, `ratio_prep` and `focmec_prep`.
