"""Tests for dtype validation before entropy estimation."""

import jax.numpy as jnp
import numpy as np
import pytest

from hoi.core.entropies import prepare_for_it
from hoi.metrics import Oinfo


@pytest.mark.parametrize("method", ["gc", "knn", "kernel", "histogram"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("container", [np.asarray, jnp.asarray])
def test_continuous_float_categories(method, dtype, container):
    values = (
        np.random.default_rng(42).standard_normal((1, 3, 64)).astype(dtype)
    )
    prepared, _ = prepare_for_it(container(values), method)
    assert prepared.shape == values.shape
    assert np.isfinite(prepared).all()


@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ],
)
@pytest.mark.parametrize("container", [np.asarray, jnp.asarray])
def test_discrete_integer_categories(dtype, container):
    values = (np.arange(192).reshape(1, 3, 64) % 4).astype(dtype)
    prepared, _ = prepare_for_it(container(values), "binning")
    np.testing.assert_array_equal(np.asarray(prepared), values)


@pytest.mark.parametrize("method", ["gc", "knn", "kernel", "histogram"])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.bool_, np.complex64, str, object]
)
def test_continuous_invalid_categories_still_rejected(method, dtype):
    values = np.ones((1, 3, 64)).astype(dtype)
    with pytest.raises(ValueError, match="data dtype should be float"):
        prepare_for_it(values, method)


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.bool_, np.complex64, str, object]
)
def test_discrete_invalid_categories_still_rejected(dtype):
    values = np.ones((1, 3, 64)).astype(dtype)
    with pytest.raises(ValueError, match="data dtype should be integer"):
        prepare_for_it(values, "binning")


@pytest.mark.parametrize(
    "method,dtypes",
    [
        ("gc", [np.float32, np.float64]),
        ("binning", [np.int32, np.int64, np.uint32, np.uint64]),
    ],
)
def test_public_oinfo_matches_across_equivalent_dtypes(method, dtypes):
    if method == "gc":
        values = (
            np.random.default_rng(42)
            .standard_normal((64, 3))
            .astype(np.float32)
        )
    else:
        values = np.tile(
            np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]), (16, 1)
        )
    outputs = [
        Oinfo(values.astype(dtype), verbose=False).fit(
            minsize=3, maxsize=3, method=method
        )
        for dtype in dtypes
    ]
    for value in outputs[1:]:
        np.testing.assert_allclose(value, outputs[0], atol=1e-6, rtol=0)


def test_sample_selection_and_gc_kwargs_unchanged():
    values = (
        np.random.default_rng(42)
        .standard_normal((1, 3, 64))
        .astype(np.float32)
    )
    prepared, kwargs = prepare_for_it(
        values, "gc", samples=np.arange(32), biascorrect=False
    )
    assert prepared.shape == (1, 3, 32)
    assert kwargs == {"biascorrect": False, "copnorm": False}
