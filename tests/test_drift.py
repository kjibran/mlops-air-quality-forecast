import numpy as np
import pandas as pd

from mlops_air_quality_forecast.drift import psi

rng = np.random.default_rng(42)


def test_same_distribution_is_stable():
    a = pd.Series(rng.normal(10, 3, 5000))
    b = pd.Series(rng.normal(10, 3, 5000))
    assert psi(a, b) < 0.1


def test_shifted_distribution_is_significant():
    a = pd.Series(rng.normal(10, 3, 5000))
    b = pd.Series(rng.normal(14, 3, 5000))
    assert psi(a, b) > 0.25
