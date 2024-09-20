"""
Functions to compute entropies.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import digamma as psi
from jax.scipy.special import gamma, ndtri
from jax.scipy.stats import gaussian_kde

from hoi.utils.logging import logger
from hoi.utils.stats import normalize

###############################################################################
###############################################################################
#                                 SWITCHER
###############################################################################
###############################################################################


def get_entropy(method="gc", **kwargs):
    """Get entropy function.

    Parameters
    ----------
    method : {'gc', 'gauss', 'binning', 'histogram', 'knn', 'kernel'}
        Name of the method to compute entropy. Use either :

            * 'gc': gaussian copula entropy [default]. See
                :func:`hoi.core.entropy_gc`
            * 'gauss': gaussian entropy. See :func:`hoi.core.entropy_gauss`
            * 'binning': estimator to use for discrete variables. See
                :func:`hoi.core.entropy_bin`
            * 'histogram' : estimator based on binning the data, to estimate
                the probability distribution of the variables and then
                compute the differential entropy. For more details see
                :func:`hoi.core.entropy_hist`
            * 'knn': k-nearest neighbor estimator. See
                :func:`hoi.core.entropy_knn`
            * 'kernel': kernel-based estimator of entropy
                see :func:`hoi.core.entropy_kernel`
            * A custom entropy estimator can be provided. It should be a
                callable function written with Jax taking a single 2D input
                of shape (n_features, n_samples) and returning a float.

    kwargs : dict | {}
        Additional arguments sent to the entropy function.

    Returns
    -------
    fcn : callable
        Function to compute entropy on a variable of shape
        (n_features, n_samples)
    """
    if method == "gc":
        return partial(entropy_gc, **kwargs)
    elif method == "gauss":
        return entropy_gauss
    elif method == "binning":
        return partial(entropy_bin, **kwargs)
    elif method == "histogram":
        return partial(entropy_hist, **kwargs)
    elif method == "knn":
        return partial(entropy_knn, **kwargs)
    elif method == "kernel":
        return partial(entropy_kernel, **kwargs)
    elif callable(method):
        # test the function
        try:
            assert method(np.random.rand(2, 100)).shape == ()
        except Exception:
            import traceback

            logger.error(traceback.format_exc())

            raise AssertionError(
                "A custom estimator should be a callable function written in "
                "Jax and taking as an input a variable x of shape (n_features,"
                " n_samples) and returning the entropy of the variable as a "
                "float."
            )

        # jit the function
        return jax.jit(method)
    else:
        raise ValueError(f"Method {method} doesn't exist.")


###############################################################################
###############################################################################
#                             PREPROCESSING
###############################################################################
###############################################################################


def prepare_for_it(data, method, samples=None, **kwargs):
    """Prepare the data before computing entropy."""
    # data.shape = n_variables, n_features, n_samples

    # -------------------------------------------------------------------------
    # type checking
    if (method in ["binning"]) and (data.dtype != int):
        raise ValueError(
            "data dtype should be integer. Check that you discretized your"
            " data. If so, use `data.astype(int)`"
        )
    elif (method in ["kernel", "gc", "knn", "histogram"]) and (
        data.dtype != float
    ):
        raise ValueError(f"data dtype should be float, not {data.dtype}")

    # -------------------------------------------------------------------------
    # trial selection
    if isinstance(samples, (np.ndarray, jnp.ndarray, list, tuple)):
        logger.info("    Sample selection")
        data = data[..., samples]
    data = jnp.asarray(data)

    # -------------------------------------------------------------------------
    # method specific preprocessing
    if method == "gc":
        logger.info("    Copnorm and demean the data")
        data = preproc_gc_3d(data)
        kwargs["copnorm"] = False
    # elif method == "kernel":
    #     logger.info("    Unit circle normalization")
    #     data = preproc_kernel_3d(data)

    return data, kwargs


###############################################################################
###############################################################################
#                              GAUSSIAN COPULA
###############################################################################
###############################################################################


@partial(jax.jit, static_argnums=(1, 2))
def entropy_gc(
    x: jnp.array, biascorrect: bool = True, copnorm: bool = True
) -> jnp.array:
    """Gaussian Copula entropy.

    Parameters
    ----------
    x : array_like
        Array of data of shape (n_features, n_samples)
    biascorrect : bool | True
        Specifies whether bias correction should be applied to the estimated MI
    copnorm : bool | True
        Apply gaussian copula normalization

    Returns
    -------
    hx : float
        Entropy of x (in bits)
    """
    nfeat, nsamp = x.shape

    # copula normalization
    if copnorm:
        x = preproc_gc_2d(x)

    # covariance
    c = jnp.dot(x, x.T) / float(nsamp - 1)
    chc = jnp.linalg.cholesky(c)

    # entropy in nats
    hx = jnp.sum(jnp.log(jnp.diagonal(chc))) + 0.5 * nfeat * (
        jnp.log(2 * jnp.pi) + 1.0
    )

    ln2 = jnp.log(2)
    if biascorrect:
        psiterms = (
            psi((nsamp - jnp.arange(1, nfeat + 1).astype(float)) / 2.0) / 2.0
        )
        dterm = (ln2 - jnp.log(nsamp - 1.0)) / 2.0
        hx = hx - nfeat * dterm - psiterms.sum()

    # convert to bits
    return hx / ln2


@jax.jit
def _preproc_gc(x: jnp.array) -> jnp.array:
    """Preprocessing for the Gaussian copula entropy and mutual-information.

    This function performs the two following steps on a vector :

        * Apply the copula normalization
        * Demean the copnormed data

    Parameters
    ----------
    x : array_like
        Array of data of shape (n_samples,)

    Returns
    -------
    cx : array_like
        Standard normal samples with the same empirical CDF value as the input.
    """
    # compute the empirical CDF
    cdf = jnp.argsort(jnp.argsort(x)).astype(float)
    cdf += 1.0
    cdf /= float(cdf.shape[-1] + 1)

    # infer normal distribution giving this CDF
    gauss = ndtri(cdf)

    # demean the gaussian
    gauss -= gauss.mean()

    return gauss


# preprocessing for a 2D variable (n_features, n_samples)
preproc_gc_2d = jax.jit(jax.vmap(_preproc_gc, in_axes=0))

# preprocessing for a 3D variable (n_variables, n_features, n_samples)
preproc_gc_3d = jax.jit(jax.vmap(preproc_gc_2d, in_axes=0))


###############################################################################
###############################################################################
#                                  GAUSSIAN
###############################################################################
###############################################################################


def entropy_gauss(x: jnp.array) -> jnp.array:
    """Gaussian entropy.

    Parameters
    ----------
    x : array_like
        Array of data of shape (n_features, n_samples)

    Returns
    -------
    hx : float
        Entropy of x (in bits)
    """
    return entropy_gc(x, biascorrect=False, copnorm=False)


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
        Entropy of x (in bits)
    """

    n_features, n_samples = x.shape

    # here, we count the number of possible multiplets. The worst is that each
    # trial is unique. So we can prepare the output to be at most (n_samples,)
    # and if trials are repeated, just set to zero it's going to be compensated
    # by the entr() function
    counts = jnp.unique(
        x, return_counts=True, size=n_samples, axis=1, fill_value=0
    )[1]
    probs = counts / n_samples

    return jax.scipy.special.entr(probs).sum() / jnp.log(base)


###############################################################################
###############################################################################
#                               HISTOGRAM
###############################################################################
###############################################################################


@partial(jax.jit, static_argnums=(1, 2))
def entropy_hist(x: jnp.array, base: float = 2, n_bins: int = 8) -> jnp.array:
    """Entropy using binning.

    Parameters
    ----------
    x : array_like
        Input data of shape (n_features, n_samples). The data should already
        be discretize
    base : float | 2
        The logarithmic base to use. Default is base 2.
    n_bins : int | 8
        The number of bin to be considered in the binarization process

    Returns
    -------
    hx : float
        Entropy of x (in bits)
    """

    # bin size computation
    bins_arr = (x.max(axis=1) - x.min(axis=1)) / n_bins
    bin_s = jnp.prod(bins_arr)

    # binning of the data
    x_min, x_max = x.min(axis=1, keepdims=True), x.max(axis=1, keepdims=True)
    dx = (x_max - x_min) / n_bins
    x_binned = ((x - x_min) / dx).astype(int)
    x_binned = jnp.minimum(x_binned, n_bins - 1).astype(int)

    n_features, n_samples = x_binned.shape

    # here, we count the number of possible multiplets. The worst is that each
    # trial is unique. So we can prepare the output to be at most (n_samples,)
    # and if trials are repeated, just set to zero it's going to be compensated
    # by the entr() function

    counts = jnp.unique(
        x_binned, return_counts=True, size=n_samples, axis=1, fill_value=0
    )[1]

    probs = counts / n_samples

    return bin_s * jax.scipy.special.entr(probs / bin_s).sum() / jnp.log(base)


###############################################################################
###############################################################################
#                                    KNN
###############################################################################
###############################################################################


@partial(jax.jit, static_argnums=(1,))
def entropy_knn(x, k: int = 3) -> jnp.array:
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
        Entropy of x (in bits)
    """
    # x = jnp.atleast_2d(x)
    d, n = float(x.shape[0]), float(x.shape[1])
    # compute euclidian distance
    x = x.T[None]
    diff = x.transpose(1, 0, 2) - x
    eucl_xi = jnp.sqrt(jnp.sum(diff**2, axis=-1))
    # dist to kth neighbor
    dist_k = jnp.sort(eucl_xi, axis=-1)[:, k]

    # volume of unit ball in d^n
    c_d = (jnp.pi ** (d * 0.5)) / gamma(1.0 + d * 0.5) / (2**d)
    log_c_d = jnp.log(c_d)

    # sum log of distances
    sum_log_dist = jnp.sum(jnp.log(2 * dist_k))

    h = -psi(k) + psi(n) + log_c_d + (d / n) * sum_log_dist

    return jnp.maximum(0, h) / jnp.log(2)  # added compared to original code


###############################################################################
###############################################################################
#                                  KERNEL
###############################################################################
###############################################################################


@partial(jax.jit, static_argnums=(1,))
def entropy_kernel(x: jnp.array, bw_method: str = None) -> jnp.array:
    """Entropy using gaussian kernel density.

    Parameters
    ----------
    x : array_like
        Input data of shape (n_features, n_samples)
    bw_method : str | None
        Estimator bandwidth. See jax.scipy.stats.gaussian_kde.

    Returns
    -------
    hx : float
        Entropy of x (in bits)
    """
    model = gaussian_kde(x, bw_method=bw_method)
    return -jnp.mean(jnp.log2(model(x)))
    # p = model.pdf(x)
    # return jax.scipy.special.entr(p).sum() / np.log(base)


# kernel preprocessing for a 2d variable (n_features, n_samples)
preproc_kernel_2d = jax.jit(
    jax.vmap(partial(normalize, to_min=-1.0, to_max=1.0), in_axes=0)
)

# kernel preprocessing for a 2d variable (n_variables, n_features, n_samples)
preproc_kernel_3d = jax.jit(jax.vmap(preproc_kernel_2d, in_axes=0))
