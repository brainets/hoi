import pytest
import numpy as np
import jax
from hoi.core.entropies import (
    entropy_gcmi,
    entropy_bin,
    entropy_knn,
    entropy_kernel,
    copnorm_nd,
    ctransform,
)
from hoi.utils import digitize

x1 = np.random.rand(1, 50)
x2 = np.random.rand(10, 50)
# l1 = [[np.random.random() for _ in range(1)] for _ in range(50)]
# l2 = [[np.random.random() for _ in range(10)] for _ in range(50)]
j1 = jax.random.uniform(jax.random.PRNGKey(0), shape=(1, 50))
j2 = jax.random.uniform(jax.random.PRNGKey(0), shape=(10, 50))


# Smoke tests
@pytest.mark.parametrize("x", [x1, x2, j1, j2])
@pytest.mark.parametrize("biascorrect", [True, False])
@pytest.mark.parametrize("demean", [True, False])
def test_entropy_gcmi(x, biascorrect, demean):
    hx = entropy_gcmi(x, biascorrect, demean)
    hx = np.asarray(hx)
    assert hx.dtype == np.float32
    pass


@pytest.mark.parametrize("x", [x1, x2, j1, j2])
def test_entropy_bin(x):
    x_bin = digitize(x, n_bins=3)
    hx = entropy_bin(x_bin)
    hx = np.asarray(hx)
    assert hx.dtype == np.float32
    pass


@pytest.mark.parametrize("x", [x1, x2, j1, j2])
@pytest.mark.parametrize(
    "base", [np.random.randint(1, 100) for _ in range(10)]
)
def test_entropy_kernel(x, base):
    hx = entropy_kernel(x, base)
    hx = np.asarray(hx)
    assert hx.dtype == np.float32


@pytest.mark.parametrize("x", [x1, x2, j1, j2])
def test_entropy_knn(x):
    hn = entropy_knn(x)
    assert hn.dtype == np.float32


# tests core/entropies/copnorm_nd.py
@pytest.mark.parametrize("x", [x1, x2])
def test_copnorm_nd(x):
    cx = copnorm_nd(x)
    assert isinstance(cx, np.ndarray)
    assert cx.shape == x.shape


@pytest.mark.parametrize("x", [x1, x2])
def test_ctransform(x):
    xr = ctransform(x)
    assert isinstance(xr, np.ndarray)
    assert xr.shape == x.shape
    for row in xr:
        for cdf in row:
            assert cdf >= 0 and cdf <= 1
