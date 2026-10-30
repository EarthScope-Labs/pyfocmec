import os
import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures():
    return FIXTURES


@pytest.fixture
def workdir(tmp_path):
    """A scratch directory holding copies of all fixture input files."""
    for f in (FIXTURES / "inputs").iterdir():
        shutil.copy(f, tmp_path)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(cwd)
