import pytest

import numpy as np
import jax.numpy as jnp

from math import comb as ccomb
from hoi.core.combinatory import combinations, _combinations
from collections.abc import Iterable


class TestCombinatory(object):
    @pytest.mark.parametrize("target", [[], [20], [20, 21]])
    @pytest.mark.parametrize(
        "n", [np.random.randint(5, 10) for _ in range(10)]
    )
    @pytest.mark.parametrize(
        "k", [np.random.randint(5, 10) for _ in range(10)]
    )
    @pytest.mark.parametrize("order", [True, False])
    def test_single_combinations(self, n, k, order, target):
        c = list(_combinations(n, k, order, target))

        # test that the number of combinations is correct
        assert len(c) == ccomb(n, k)

        # check the order
        if order:
            assert all([o == k + len(target) for o in c])
        else:
            assert all([len(o) == k + len(target) for o in c])

        # check that targets are included
        if len(target) and not order:
            assert all([all([m in o for m in target]) for o in c])

    @pytest.mark.parametrize("fill", [-1, -10])
    @pytest.mark.parametrize("target", [None, [20], [20, 21]])
    @pytest.mark.parametrize("order", [True, False])
    @pytest.mark.parametrize("astype", ["numpy", "jax", "iterator"])
    @pytest.mark.parametrize(
        "max", [_ for _ in range(2)]
    )  # addition to minimum size
    @pytest.mark.parametrize(
        "min", [np.random.randint(1, 10) for _ in range(2)]
    )
    @pytest.mark.parametrize("n", [np.random.randint(5, 10) for _ in range(2)])
    def test_combinations(self, n, min, max, astype, order, target, fill):
        # get combinations
        combs = combinations(
            n,
            min,
            maxsize=min + max,
            astype=astype,
            order=order,
            target=target,
            fill_value=fill,
        )

        # check the number of multiplets
        n_mults = 0
        for c in range(min, min + max + 1):
            n_mults += ccomb(n, c)
        if astype in ["jax", "numpy"]:
            assert combs.shape[0] == n_mults
        elif astype == "iterator":
            assert len([c for c in combs]) == n_mults

        # check type
        if astype == "numpy":
            assert isinstance(combs, np.ndarray)
        elif astype == "jax":
            assert isinstance(combs, jnp.ndarray)
        elif astype == "iterator":
            assert isinstance(combs, Iterable)


if __name__ == "__main__":
    # TestCombinatory().test_single_combinations(5, 3, False, [21, 22])
    TestCombinatory().test_combinations(10, 2, 3, "iterator", False, None, -1)
