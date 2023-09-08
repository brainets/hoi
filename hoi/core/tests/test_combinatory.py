import pytest
import numpy as np
from math import comb as ccomb
from hoi.core.combinatory import combinations, _combinations
from collections.abc import Iterable


class TestCombinatory(object):
    @pytest.mark.parametrize(
        "n", [np.random.randint(5, 10) for _ in range(10)]
    )
    @pytest.mark.parametrize(
        "k", [np.random.randint(5, 10) for _ in range(10)]
    )
    @pytest.mark.parametrize("order", [True, False])
    def test_single_combinations(self, n, k, order):
        c = list(_combinations(n, k, order))
        assert len(c) == ccomb(n, k)
        pass

    @pytest.mark.parametrize("n", [np.random.randint(5, 10) for _ in range(2)])
    @pytest.mark.parametrize(
        "min", [np.random.randint(1, 10) for _ in range(2)]
    )
    @pytest.mark.parametrize(
        "max", [_ for _ in range(2)]
    )  # addition to minimum size
    @pytest.mark.parametrize("astype", ["numpy", "jax", "iterator"])
    @pytest.mark.parametrize("order_val", [True, False])
    def test_combinations(self, n, min, max, astype, order_val):
        combs = combinations(n, min, min + max, astype, order_val)
        assert isinstance(combs, Iterable)
        pass
