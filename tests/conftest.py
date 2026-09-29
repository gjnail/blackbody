import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope='session')
def engine():
    try:
        from blackbody.engine.engine import Engine
        return Engine()
    except Exception as ex:  # no GPU on this machine
        pytest.skip(f'GPU engine unavailable: {ex}')
