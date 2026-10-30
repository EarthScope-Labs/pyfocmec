"""``pyfocmec`` umbrella command."""

from __future__ import annotations

import sys

from .. import __version__

_COMMANDS = {
    "search": ("focmec", "main", "grid search for focal mechanisms (same as the `focmec` command)"),
    "dsretc": ("dsretc", "main", "convert between DSR, P/T axes and moment tensors; get the focmec Angle"),
    "radiat": ("tools", "radiat_main", "P/SV/SH radiation factors of a double couple"),
    "freesurf": ("tools", "freesurf_main", "free-surface amplitude corrections"),
    "synth": ("tools", "synth_main", "synthetic FOCMEC input for a known mechanism"),
    "convert": ("tools", "convert_main", "convert a pre-2017 FOCMEC input file"),
}


def _extend_commands():
    for name, spec in (
        ("ratio-prep", ("prep", "ratio_prep_main", "hypocentre + stations -> angles and travel times")),
        ("prep", ("prep", "focmec_prep_main", "picks + ratio-prep table -> FOCMEC input file")),
        ("hypo2foc", ("prep", "hypo2foc_main", "hypoellipse archive -> FOCMEC input files")),
        ("focplt", ("plot", "focplt_main", "plot polarities and solutions on the focal sphere")),
        ("ratplt", ("plot", "ratplt_main", "plot amplitude-ratio misfit on the focal sphere")),
    ):
        _COMMANDS[name] = spec


_extend_commands()


def _available():
    import importlib

    out = {}
    for name, (mod, func, desc) in _COMMANDS.items():
        try:
            m = importlib.import_module(f"{__package__}.{mod}")
        except ModuleNotFoundError as exc:
            if exc.name != f"{__package__}.{mod}":
                raise  # a missing dependency inside the module is a real problem, not an absent command
            continue
        if hasattr(m, func):
            out[name] = (getattr(m, func), desc)
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = _available()
    if not argv or argv[0] in ("-h", "--help"):
        print(f"pyfocmec {__version__} - tools for the FOCMEC focal-mechanism package\n")
        print("usage: pyfocmec <command> [options]\n\ncommands:")
        for name, (_, desc) in cmds.items():
            print(f"  {name:<11s} {desc}")
        print("\nRun `pyfocmec <command> --help` for details.")
        return 0
    if argv[0] == "--version":
        print(f"pyfocmec {__version__}")
        return 0
    if argv[0] not in cmds:
        print(f"pyfocmec: unknown command {argv[0]!r}; try `pyfocmec --help`", file=sys.stderr)
        return 2
    return cmds[argv[0]][0](argv[1:]) or 0


if __name__ == "__main__":
    sys.exit(main())
