"""pyfocmec: Python port of FOCMEC (Arthur Snoke, Virginia Tech).

Focal mechanisms of double-couple sources from P, SV and SH polarities and
SV/P, SH/P and SV/SH amplitude ratios, using a systematic grid search of the
focal sphere.
"""

__version__ = "0.1.0"

from .inp import FocmecData, parse_inp, read_inp
from .params import FocmecParams
from .search import FocmecResult, Solution, search

__all__ = [
    "FocmecData",
    "FocmecParams",
    "FocmecResult",
    "Solution",
    "parse_inp",
    "read_inp",
    "search",
    "__version__",
]
