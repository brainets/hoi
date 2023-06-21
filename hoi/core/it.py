from functools import partial
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from scipy.special import ndtri
import logging
from jax.scipy.special import digamma as psi

logger = logging.getLogger("frites")


@partial(jax.jit, static_argnums=1)
def ent_tensor_g(x: jnp.array, biascorrect=True) -> jnp.array:
    """Entropy of a tensor of shape (..., n_vars, n_trials)
    Parameters
    ----------
    x : jnp.array
        Input tensor of shape (..., n_vars, n_trials).
    biascorrect : bool, optional
        Whether to apply bias correction. Default is True.
    Returns
    -------
    jnp.array
        Entropy of the input tensor in nats which is is a measure of the 
        uncertainty or disorder of the input tensor x.
    """
    nvarx, ntrl = x.shape[-2], x.shape[-1]

    # covariance
    c = jnp.einsum("...ij, ...kj->...ik", x, x)
    c /= float(ntrl - 1.0)
    chc = jnp.linalg.cholesky(c)

    # entropy in nats
    hx = jnp.log(jnp.einsum("...ii->...i", chc)).sum(-1) + 0.5 * nvarx * (
        jnp.log(2 * jnp.pi) + 1.0
    )

    ln2 = jnp.log(2)
    if biascorrect:
        psiterms = (
            jsp.digamma((ntrl - jnp.arange(1, nvarx + 1).astype(float)) / 2.0) / 2.0
        )
        dterm = (ln2 - jnp.log(ntrl - 1.0)) / 2.0
        hx = hx - nvarx * dterm - psiterms.sum()

    return hx


@partial(jax.jit, static_argnums=1)
def ent_vector_g(x: jnp.array, biascorrect: bool = True) -> jnp.array:
    """Entropy of an array of shape (n_features, n_samples)."""
    nvarx, ntrl = x.shape

    # demean data
    # x = x - x.mean(axis=1, keepdims=True)

    # covariance
    c = jnp.dot(x, x.T) / float(ntrl - 1)
    chc = jnp.linalg.cholesky(c)

    # entropy in nats
    hx = jnp.sum(jnp.log(jnp.diagonal(chc))) + 0.5 * nvarx * (jnp.log(2 * jnp.pi) + 1.0)

    ln2 = jnp.log(2)
    if biascorrect:
        psiterms = psi((ntrl - jnp.arange(1, nvarx + 1).astype(float)) / 2.0) / 2.0
        dterm = (ln2 - jnp.log(ntrl - 1.0)) / 2.0
        hx = hx - nvarx * dterm - psiterms.sum()

    # convert to bits
    return hx / ln2


def ctransform(x):  # frites.core
    xr = np.argsort(np.argsort(x)).astype(float)
    xr += 1.0
    xr /= float(xr.shape[-1] + 1)
    return xr


def copnorm_1d(x):  # frites.core
    assert isinstance(x, np.ndarray) and (x.ndim == 1)
    return ndtri(ctransform(x))


def copnorm_nd(x, axis=-1):  # frites.core
    assert isinstance(x, np.ndarray) and (x.ndim >= 1)
    return np.apply_along_axis(copnorm_1d, axis, x)
