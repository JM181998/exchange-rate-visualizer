import numpy as np
import pandas as pd
import pytest

from fxviz import config


@pytest.fixture
def synthetic_rates():
    """~8 years of daily random-walk rates for the four pairs."""
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2017-01-02", periods=2000, name="Date")
    cols = {}
    for i, pair in enumerate(config.PAIRS):
        cols[pair] = (1.0 + 0.2 * i) * np.exp(np.cumsum(rng.normal(0, 0.004, len(index))))
    return pd.DataFrame(cols, index=index)
