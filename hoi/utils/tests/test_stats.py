import pytest
import numpy as np
import jax
from hoi.utils.stats import digitize, landscape, normalize, get_nbest_mult
import pandas as pd
import math
import xarray as xr
from hoi.metrics import Oinfo

x1 = np.random.rand(1, 50)
x2 = np.random.rand(10, 50)
j1 = jax.random.uniform(jax.random.PRNGKey(0), shape=(1, 50))
j2 = jax.random.uniform(jax.random.PRNGKey(0), shape=(10, 50))


def truncate_decimal(number, decimal_points):
    factor = 10**decimal_points
    truncated_number = math.floor(number * factor) / factor
    return truncated_number


# test landscape
x = [np.random.randint(5, size=(10, 50)) for i in range(3)]
multi = [np.random.randint(5, size=(10, 50)) for j in range(3)]


# tests digitze_1d, digitize_sklearn, digitize
class TestStats(object):
    @pytest.mark.parametrize("arr", [x2, j2])
    @pytest.mark.parametrize("bins", [n + 2 for n in range(5)])
    @pytest.mark.parametrize("sklearn", [True, False])
    def test_digitize(self, arr, bins, sklearn):
        x_binned = digitize(x=arr, n_bins=bins, axis=0, use_sklearn=sklearn)
        assert arr.shape == x_binned.shape
        for row in x_binned:
            for val in row:
                assert isinstance(val, np.int64)

    @pytest.mark.parametrize("x", [x1, x2, j2])
    @pytest.mark.parametrize(
        "to_min", [np.random.uniform(0, 1) for n in range(5)]
    )
    def test_normalize(self, x, to_min):
        to_max = to_min + np.random.uniform(0, 1)
        to_max = to_min + np.random.uniform(0, 1)
        xn = normalize(x, to_min, to_max)
        assert xn.shape == x.shape
        assert np.min(xn) >= truncate_decimal(to_min, 3)
        assert np.max(xn) <= to_max

    @pytest.mark.parametrize("x", x)
    @pytest.mark.parametrize("multi", multi)
    @pytest.mark.parametrize(
        "n_bins", [np.random.randint(100, 500) for _ in range(5)]
    )
    def test_landscape(self, x, multi, n_bins):
        op = landscape(x, multi, n_bins, output="numpy")
        assert op[0].shape == (n_bins, 5)
        assert isinstance(op[0], np.ndarray)
        op = landscape(x, multi, n_bins, output="pandas")
        assert op.shape == (n_bins, 5)
        assert isinstance(op, pd.DataFrame)
        op = landscape(x, multi, n_bins, output="xarray")
        assert op.shape == (n_bins, 5)
        assert isinstance(op, xr.DataArray)

    # test get_nbest_mult
    @pytest.mark.parametrize("x", [np.random.rand(100, 5) for _ in range(3)])
    @pytest.mark.parametrize(
        "n_best", [np.random.randint(6, 20) for _ in range(3)]
    )
    def test_nbest(self, x, n_best):
        model = Oinfo(x)
        hoi = model.fit()
        df = get_nbest_mult(hoi, model=model, n_best=n_best)
        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 2 * n_best
        pass
