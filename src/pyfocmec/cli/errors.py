"""One-line error reporting for the command-line tools."""

from __future__ import annotations

import functools
import os
import sys

_HINTS = {
    "matplotlib": "plotting needs matplotlib: pip install 'pyfocmec[plot]'",
    "obspy": "this needs obspy: pip install 'pyfocmec[taup]'",
}


def clean_errors(prog: str):
    """Report expected failures (bad input, missing file or optional dependency) as one line.

    Returns exit status 1.  Set ``PYFOCMEC_DEBUG=1`` to get the traceback instead.
    """

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (OSError, ValueError, KeyError, ImportError) as exc:
                if os.environ.get("PYFOCMEC_DEBUG"):
                    raise
                if isinstance(exc, ImportError) and getattr(exc, "name", None) in _HINTS:
                    msg = _HINTS[exc.name]
                elif isinstance(exc, KeyError) and exc.args:
                    msg = exc.args[0]
                elif isinstance(exc, FileNotFoundError) and exc.filename:
                    msg = f"cannot open {exc.filename}: no such file"
                else:
                    msg = exc
                print(f"{prog}: error: {msg}", file=sys.stderr)
                return 1

        return wrapper

    return decorate
