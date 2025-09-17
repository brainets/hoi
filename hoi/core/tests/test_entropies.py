import jax
import numpy as np
import pytest

from hoi.core.entropies import (
    entropy_bin,
    entropy_gauss,
    entropy_gc,
    entropy_hist,
    entropy_kernel,
    entropy_knn,
    get_entropy,
)
from hoi.utils import digitize

x1 = np.random.rand(1, 50)
x2 = np.random.rand(10, 50)

j1 = jax.random.uniform(jax.random.PRNGKey(0), shape=(1, 50))
j2 = jax.random.uniform(jax.random.PRNGKey(0), shape=(10, 50))


# custom estimator
def custom_est(x):
    return x.mean()


# method names
names = ["gc", "gauss", "knn", "kernel", "binning", "histogram", custom_est]


# Smoke tests
class TestEntropy(object):
    @pytest.mark.parametrize("x", [x1, x2, j1, j2])
    @pytest.mark.parametrize("biascorrect", [True, False])
    @pytest.mark.parametrize("copnorm", [True, False])
    def test_entropy_gc(self, x, biascorrect, copnorm):
        hx = entropy_gc(x, biascorrect=biascorrect, copnorm=copnorm)
        hx = np.asarray(hx)
        assert hx.dtype == np.float32
        assert hx.shape == ()

    @pytest.mark.parametrize("x", [x1, x2, j1, j2])
    def test_entropy_bin(self, x):
        x_bin = digitize(x, n_bins=3)
        hx = entropy_bin(x_bin)
        hx = np.asarray(hx)
        assert hx.dtype == np.float32
        assert hx.shape == ()

    @pytest.mark.parametrize("x", [x1, x2, j1, j2])
    def test_entropy_hist(self, x):
        hx = entropy_hist(x)
        hx = np.asarray(hx)
        assert hx.dtype == np.float32
        assert hx.shape == ()

    @pytest.mark.parametrize("x", [x1, x2, j1, j2])
    def test_entropy_kernel(self, x):
        hx = entropy_kernel(x)
        hx = np.asarray(hx)
        assert hx.dtype == np.float32
        assert hx.shape == ()

    @pytest.mark.parametrize("x", [x1, x2, j1, j2])
    def test_entropy_gauss(self, x):
        hx = entropy_gauss(x)
        hx = np.asarray(hx)
        assert hx.dtype == np.float32
        assert hx.shape == ()

    @pytest.mark.parametrize("x", [x1, x2, j1, j2])
    @pytest.mark.parametrize("k", [2, 10])
    def test_entropy_knn(self, x, k):
        hn = entropy_knn(x, k=k)
        assert hn.dtype == np.float32
        assert hn.shape == ()

    @pytest.mark.parametrize("x", [x1, x2, j1, j2])
    @pytest.mark.parametrize("name", names)
    def test_get_entropy(self, name, x):
        fcn = get_entropy(method=name)
        hn = fcn(x)
        assert hn.dtype == np.float32
        assert hn.shape == ()
