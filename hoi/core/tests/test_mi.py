import pytest
import numpy as np
import jax
from hoi.core.mi import get_mi
from hoi.utils import digitize

x1 = np.random.rand(1, 50)
x2 = np.random.rand(10, 50)

j1 = jax.random.uniform(jax.random.PRNGKey(0), shape=(1, 50))
j2 = jax.random.uniform(jax.random.PRNGKey(0), shape=(10, 50))


# custom estimator
def custom_est(x, y):
    return x.mean() + y.mean()


# method names
names = ["gc", "gauss", "knn", "kernel", "binning", custom_est]


# Smoke tests
class TestMutualInformation(object):
    @pytest.mark.parametrize("xy", [(x1, x2), (j1, j2)])
    @pytest.mark.parametrize("biascorrect", [True, False])
    @pytest.mark.parametrize("copnorm", [True, False])
    def test_mi_gc(self, xy, biascorrect, copnorm):
        mi_fcn = get_mi(method="gc", biascorrect=biascorrect, copnorm=copnorm)
        mi = mi_fcn(xy[0], xy[1])
        assert mi.dtype == np.float32
        assert mi.shape == ()

    @pytest.mark.parametrize("xy", [(x1, x2), (j1, j2)])
    def test_mi_bin(self, xy):
        mi_fcn = get_mi(method="binning")
        x_binned, _ = digitize(xy[0], n_bins=3)
        y_binned, _ = digitize(xy[1], n_bins=3)
        mi = mi_fcn(x_binned, y_binned)
        assert mi.dtype == np.float32
        assert mi.shape == ()

    @pytest.mark.parametrize("xy", [(x1, x2), (j1, j2)])
    def test_mi_kernel(self, xy):
        mi_fcn = get_mi(method="kernel")
        mi = mi_fcn(xy[0], xy[1])
        assert mi.dtype == np.float32
        assert mi.shape == ()

    @pytest.mark.parametrize("xy", [(x1, x2), (j1, j2)])
    @pytest.mark.parametrize("k", [2, 10])
    def test_mi_knn(self, xy, k):
        mi_fcn = get_mi(method="knn", k=k)
        mi = mi_fcn(xy[0], xy[1])
        assert mi.dtype == np.float32
        assert mi.shape == ()

    @pytest.mark.parametrize("xy", [(x1, x2), (j1, j2)])
    def test_mi_gauss(self, xy):
        mi_fcn = get_mi(method="gauss")
        mi = mi_fcn(xy[0], xy[1])
        assert mi.dtype == np.float32
        assert mi.shape == ()

    @pytest.mark.parametrize("xy", [(x1, x2), (j1, j2)])
    @pytest.mark.parametrize("name", names)
    def test_get_mi(self, xy, name):
        mi_fcn = get_mi(method=name)
        mi = mi_fcn(xy[0], xy[1])
        assert mi.dtype == np.float32
        assert mi.shape == ()
