import pytest
import numpy as np
import jax
from hoi.core.mi import get_mi
from hoi.utils import digitize

x1 = np.random.rand(1, 50)
x2 = np.random.rand(10, 50)

j1 = jax.random.uniform(jax.random.PRNGKey(0), shape=(1, 50))
j2 = jax.random.uniform(jax.random.PRNGKey(0), shape=(10, 50))


# Smoke tests
class TestMutualInformation(object):
    @pytest.mark.parametrize("xy", [(x1, x2), (j1, j2)])
    @pytest.mark.parametrize("biascorrect", [True, False])
    @pytest.mark.parametrize("demean", [True, False])
    def test_mi_gcmi(self, xy, biascorrect, demean):
        mi_fcn = get_mi(method="gcmi", biascorrect=biascorrect, demean=demean)
        mi = mi_fcn(xy[0], xy[1])
        assert mi.dtype == np.float32
        assert mi.shape == ()

    @pytest.mark.parametrize("xy", [(x1, x2), (j1, j2)])
    @pytest.mark.parametrize("biascorrect", [True, False])
    @pytest.mark.parametrize("demean", [True, False])
    def test_mi_bin(self, xy, biascorrect, demean):
        mi_fcn = get_mi(method="gcmi", biascorrect=biascorrect, demean=demean)
        x_binned = digitize(xy[0], n_bins=3)
        y_binned = digitize(xy[1], n_bins=3)
        mi = mi_fcn(x_binned, y_binned)
        assert mi.dtype == np.float32
        assert mi.shape == ()

    @pytest.mark.parametrize("xy", [(x1, x2), (j1, j2)])
    @pytest.mark.parametrize(
        "base", [np.random.randint(1, 100) for _ in range(10)]
    )
    def test_mi_kernel(self, xy, base):
        mi_fcn = get_mi(method="kernel", base=base)
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
