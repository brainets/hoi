"""
Functions to compute entropies.
"""
from functools import partial
import logging

import numpy as np
from scipy.special import ndtri

import jax
import jax.numpy as jnp
from jax.scipy.special import digamma as psi
from jax.scipy.special import gamma
from jax.scipy.stats import gaussian_kde

logger = logging.getLogger("hoi")


###############################################################################
###############################################################################
#                                 SWITCHER
###############################################################################
###############################################################################


def get_entropy(method="gcmi", **kwargs):
    """Get entropy function.

    Parameters
    ----------
    method : {'gcmi', 'binning', 'knn'}
        Name of the method to compute entropy.
    kwargs : dict | {}
        Additional arguments sent to the entropy function.

    Returns
    -------
    fcn : callable
        Function to compute entropy on a variable of shape
        (n_features, n_samples)
    """
    if method == "gcmi":
        return partial(entropy_gcmi, **kwargs)
    elif method == "binning":
        return partial(entropy_bin, **kwargs)
    elif method == "knn":
        return partial(entropy_knn, **kwargs)
    elif method == "kernel":
        return partial(entropy_kernel, **kwargs)
    else:
        raise ValueError(f"Method {method} doesn't exist.")


###############################################################################
###############################################################################
#                             PREPROCESSING
###############################################################################
###############################################################################


def prepare_for_entropy(data, method, **kwargs):
    """Prepare the data before computing entropy.
    """
    n_samples, n_features, n_variables = data.shape

    # type checking
    if (method in ["binning"]) and (data.dtype != int):
        raise ValueError(
            "data dtype should be integer. Check that you discretized your"
            " data. If so, use `data.astype(int)`"
        )
    elif (method in ["kernel", "gcmi", "knn"]) and (data.dtype != float):
        raise ValueError(f"data dtype should be float, not {data.dtype}")

    # method specific preprocessing
    if method == "gcmi":
        logger.info("    Copnorm data")
        data = copnorm_nd(data, axis=0)
        data = data - data.mean(axis=0, keepdims=True)
        kwargs["demean"] = False
    elif method == "kernel":
        logger.info("    Unit circle normalization")
        from hoi.utils import normalize

        data = np.apply_along_axis(normalize, 0, data, to_min=-1.0, to_max=1.0)
    elif method == "binning":
        pass

    # make the data (n_variables, n_features, n_trials)
    data = jnp.asarray(data.transpose(2, 1, 0))

    return data, kwargs


###############################################################################
###############################################################################
#                            GAUSSIAN COPULA
###############################################################################
###############################################################################


@partial(jax.jit, static_argnums=(1, 2))
def entropy_gcmi(
    x: jnp.array, biascorrect: bool = False, demean: bool = False
) -> jnp.array:
    """Entropy of a Gaussian variable in bits.

    H = ent_g(x) returns the entropy of a (possibly multidimensional) Gaussian
    variable x with bias correction.

    Parameters
    ----------
    x : array_like
        Array of data of shape (n_features, n_samples)
    biascorrect : bool | False
        Specifies whether bias correction should be applied to the estimated MI
    demean : bool | False
        Specifies whether the input data have to be demeaned

    Returns
    -------
    hx : float
        Entropy of the gaussian variable (in bits)
    """
    nfeat, nsamp = x.shape

    # demean data
    if demean:
        x = x - x.mean(axis=1, keepdims=True)

    # covariance
    c = jnp.dot(x, x.T) / float(nsamp - 1)
    chc = jnp.linalg.cholesky(c)

    # entropy in nats
    # E501
    # hx = jnp.sum(jnp.log(jnp.diagonal(chc))) + 0.5 * nfeat *
    # (jnp.log(2 * jnp.pi) + 1.0)
    diag_sum = jnp.sum(jnp.log(jnp.diagonal(chc)))
    const_term = 0.5 * nfeat * (jnp.log(2 * jnp.pi) + 1.0)
    hx = diag_sum + const_term

    ln2 = jnp.log(2)
    if biascorrect:
        term = (nsamp - jnp.arange(1, nfeat + 1).astype(float)) / 2.0
        psiterms = psi(term) / 2.0
        dterm = (ln2 - jnp.log(nsamp - 1.0)) / 2.0
        hx = hx - nfeat * dterm - psiterms.sum()

    # convert to bits
    return hx / ln2


def ctransform(x):
    """Copula transformation (empirical CDF).

    Parameters
    ----------
    x : array_like
        Array of data. The trial axis should be the last one

    Returns
    -------
    xr : array_like
        Empirical CDF value along the last axis of x. Data is ranked and scaled
        within [0 1] (open interval)
    """
    xr = np.argsort(np.argsort(x)).astype(float)
    xr += 1.0
    xr /= float(xr.shape[-1] + 1)
    return xr


def copnorm_1d(x):
    """Copula normalization for a single vector.

    Parameters
    ----------
    x : array_like
        Array of data of shape (n_epochs,)

    Returns
    -------
    cx : array_like
        Standard normal samples with the same empirical CDF value as the input.
    """
    assert isinstance(x, np.ndarray) and (x.ndim == 1)
    return ndtri(ctransform(x))


def copnorm_nd(x, axis=-1):
    """Copula normalization for a multidimentional array.

    Parameters
    ----------
    x : array_like
        Array of data
    axis : int | -1
        Epoch (or trial) axis. By default, the last axis is considered

    Returns
    -------
    cx : array_like
        Standard normal samples with the same empirical CDF value as the input.
    """
    assert isinstance(x, np.ndarray) and (x.ndim >= 1)
    return np.apply_along_axis(copnorm_1d, axis, x)


###############################################################################
###############################################################################
#                               BINNING
###############################################################################
###############################################################################


@partial(jax.jit, static_argnums=(1,))
def entropy_bin(x: jnp.array, base: int = 2) -> jnp.array:
    """Entropy using binning.

    Parameters
    ----------
    x : array_like
        Input data of shape (n_features, n_samples). The data should already
        be discretize
    base : int | 2
        The logarithmic base to use. Default is base 2.

    Returns
    -------
    hx : float
        Entropy of x
    """
    n_features, n_samples = x.shape
    # here, we count the number of possible multiplets. The worst is that each
    # trial is unique. So we can prepare the output to be at most (n_samples,)
    # and if trials are repeated, just set to zero it's going to be compensated
    # by the entr() function
    # E501
    # counts = jnp.unique(x, return_counts=True, size=n_samples,
    # axis=1, fill_value=0)[1]
    counts = jnp.unique(x, False, False, True, 1, n_samples, 0)[1]
    probs = counts / n_samples
    return jax.scipy.special.entr(probs).sum() / np.log(base)


###############################################################################
###############################################################################
#                                    KNN
###############################################################################
###############################################################################


@partial(jax.jit)
def set_to_inf(x, _):
    """Set to infinity the minimum in a vector."""
    x = x.at[jnp.argmin(x)].set(jnp.inf)
    return x, jnp.nan


@partial(jax.jit, static_argnums=(2,))
def cdistk(xx, idx, k=1):
    """K-th minimum euclidian distance."""
    x, y = xx[:, [idx]], xx

    # compute euclidian distance
    eucl = jnp.sqrt(jnp.sum((x - y) ** 2, axis=0))

    # in case of 0-distances, replace them by infinity
    eucl = jnp.where(eucl == 0, jnp.inf, eucl)

    # set to inf to get k eucl
    eucl, _ = jax.lax.scan(set_to_inf, eucl, jnp.arange(k))

    return xx, eucl[jnp.argmin(eucl)]


@partial(jax.jit, static_argnums=(1,))
def entropy_knn(x: jnp.array, k: int = 1) -> jnp.array:
    """Entropy using the k-nearest neighbor.

    Original code: https://github.com/blakeaw/Python-knn-entropy/
    and references. See also Kraskov et al., Estimating mutual information,
    Phy rev, 2004


    Parameters
    ----------
    x : array_like
        Input data of shape (n_features, n_samples)
    knn : int | 1
        K-th closest point. Default is 1 (closest point)

    Returns
    -------
    hx : float
        Entropy of x
    """
    # x = jnp.atleast_2d(x)
    d, n = float(x.shape[0]), float(x.shape[1])

    # wrap with knn
    cdist = partial(cdistk, k=k)

    # compute euclidian distance
    _, r_k = jax.lax.scan(cdist, x, jnp.arange(int(n)).astype(int))

    # log of the volume of unit ball in d^n
    log_c_d = (d / 2.0) * jnp.log(jnp.pi) - jnp.log(
        gamma(1 + d / 2.0)
    )  # + d * jnp.log(2)

    # sum log of distances
    sum_log_dist = jnp.sum(jnp.log(2 * r_k))

    h = -psi(k) + psi(n) + log_c_d + (d / n) * sum_log_dist

    return h


###############################################################################
###############################################################################
#                                  KERNEL
###############################################################################
###############################################################################


@partial(jax.jit, static_argnums=(1, 2))
# E501
def entropy_kernel(x: jnp.array, base: 2, bw_method: str = None) -> jnp.array:
    """Entropy using gaussian kernel density.

    Parameters
    ----------
    x : array_like
        Input data of shape (n_features, n_samples)

    Returns
    -------
    hx : float
        Entropy of x
    """
    model = gaussian_kde(x, bw_method=bw_method)
    return -jnp.mean(jnp.log2(model(x)))
    # p = model.pdf(x)
    # return jax.scipy.special.entr(p).sum() / np.log(base)
