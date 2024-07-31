from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import digamma as psi

from hoi.core.entropies import get_entropy, preproc_gc_2d
from hoi.utils.logging import logger

###############################################################################
###############################################################################
#                                 SWITCHER
###############################################################################
###############################################################################


def get_mi(method="gc", **kwargs):
    """Get Mutual-Information function.

    Parameters
    ----------
    method : {'gc', 'binning', 'knn', 'kernel'}
        Name of the method to compute mutual-information.
    kwargs : dict | {}
        Additional arguments sent to the mutual-information function.

    Returns
    -------
    fcn : callable
        Function to compute mutual information on variables of shapes
        (n_features, n_samples)
    """
    if method == "gc":
        return partial(mi_gc, **kwargs)
    elif method == "gauss":
        return mi_gauss
    elif method == "knn":
        return partial(mi_knn, **kwargs)
    elif callable(method):
        # test the function
        try:
            x = np.random.rand(2, 100)
            y = np.random.rand(4, 100)
            assert method(x, y).shape == ()
        except Exception:
            import traceback

            logger.error(traceback.format_exc())

            raise AssertionError(
                "A custom estimator should be a callable function written in "
                "Jax and taking two inputs x and y of shapes (n_features_x,"
                " n_samples) and (n_features_y, n_samples) and returning the "
                "mutual information between variables as a float."
            )

        # jit the function
        return jax.jit(method)
    else:
        # get the entropy function
        _entropy = get_entropy(method=method, **kwargs)

        # wrap the mi function with it
        return partial(compute_mi, entropy_fcn=_entropy)


###############################################################################
###############################################################################
#                             PREPROCESSING
###############################################################################
###############################################################################


# def prepare_for_mi(x, y, method, **kwargs):
#     """Prepare the data before computing mutual-information."""
#     x, kwargs = prepare_for_it(x, method, **kwargs.copy())
#     return x, y, kwargs


@partial(jax.jit, static_argnums=(2))
def compute_mi_comb(inputs, comb, mi=None):
    x, y = inputs
    x_c = x[:, comb, :]
    return inputs, mi(x_c, y)


@partial(jax.jit, static_argnums=(2))
def compute_mi_comb_phi(inputs, comb, mi=None):
    x, y = inputs
    x_c = jnp.atleast_2d(x[:, comb[0], :])
    y_c = y[:, comb[1], :]

    return inputs, mi(x_c, y_c)


###############################################################################
###############################################################################
#                         GENERAL MUTUAL INFORMATION
###############################################################################
###############################################################################


@partial(jax.jit, static_argnums=(2,))
def compute_mi(x, y, entropy_fcn=None):
    """Compute the mutual-information by providing an entropy function.

    Parameters
    ----------
    x, y : array_like
        Arrays to consider for computing the Mutual Information. The two input
        variables x and y should have a shape of (n_features_x, n_samples) and
        (n_features_y, n_samples)
    entropy_fcn : function | None
        Function to use for computing the entropy.

    Returns
    -------
    mi : float
        Floating value describing the mutual-information between x and y
    """
    # compute mi
    mi = (
        entropy_fcn(x)
        + entropy_fcn(y)
        - entropy_fcn(jnp.concatenate((x, y), axis=0))
    )
    return mi


###############################################################################
###############################################################################
#                     GAUSSIAN COPULA MUTUAL INFORMATION
###############################################################################
###############################################################################


@partial(jax.jit, static_argnums=(2, 3))
def mi_gc(
    x: jnp.array,
    y: jnp.array,
    biascorrect: bool = False,
    copnorm: bool = False,
):
    """Mutual information (MI) between two Gaussian variables in bits.

    Parameters
    ----------
    x, y : array_like
        Arrays to consider for computing the Mutual Information. The two input
        variables x and y should have a shape of (n_features_x, n_samples) and
        (n_features_y, n_samples)
    biascorrect : bool | False
        Specifies whether bias correction should be applied to the estimated MI
    copnorm : bool | True
        Apply gaussian copula normalization


    Returns
    -------
    i : float
        Information shared by x and y (in bits)
    """
    n_features_x, n_samples = x.shape
    n_features_y = y.shape[0]
    n_features_xy = n_features_x + n_features_y

    # joint variable
    xy = jnp.vstack((x, y))
    if copnorm:
        xy = preproc_gc_2d(xy)
    cxy = jnp.dot(xy, xy.T) / float(n_samples - 1)
    # submatrices of joint covariance
    cx = cxy[:n_features_x, :n_features_x]
    cy = cxy[n_features_x:, n_features_x:]

    chcxy = jnp.linalg.cholesky(cxy)
    chcx = jnp.linalg.cholesky(cx)
    chcy = jnp.linalg.cholesky(cy)

    # entropies in nats
    # normalizations cancel for mutual information
    hx = jnp.sum(jnp.log(jnp.diagonal(chcx)))
    hy = jnp.sum(jnp.log(jnp.diagonal(chcy)))
    hxy = jnp.sum(jnp.log(jnp.diagonal(chcxy)))

    ln2 = jnp.log(2)
    if biascorrect:
        psiterms = (
            psi(
                (n_samples - jnp.arange(1, n_features_xy + 1)).astype(float)
                / 2.0
            )
            / 2.0
        )
        dterm = (ln2 - jnp.log(n_samples - 1.0)) / 2.0
        hx = hx - n_features_x * dterm - psiterms[:n_features_x].sum()
        hy = hy - n_features_y * dterm - psiterms[:n_features_y].sum()
        hxy = hxy - n_features_xy * dterm - psiterms[:n_features_xy].sum()

    # MI in bits
    i = (hx + hy - hxy) / ln2
    return i


###############################################################################
###############################################################################
#                       GAUSSIAN MUTUAL INFORMATION
###############################################################################
###############################################################################


def mi_gauss(x: jnp.array, y: jnp.array):
    """Mutual information (MI) between two Gaussian variables in bits.

    Parameters
    ----------
    x, y : array_like
        Arrays to consider for computing the Mutual Information. The two input
        variables x and y should have a shape of (n_features_x, n_samples) and
        (n_features_y, n_samples)

    Returns
    -------
    i : float
        Information shared by x and y (in bits)
    """
    return mi_gc(x, y, biascorrect=False, copnorm=False)


###############################################################################
###############################################################################
#                         KNN MUTUAL INFORMATION
###############################################################################
###############################################################################


@jax.jit
def _cdist(x, y) -> jnp.ndarray:
    """Pairwise squared distances between all samples of x and y."""
    diff = x.T[:, None, :] - y.T[None]
    _dist = jnp.einsum("ijc->ij", diff**2)
    return _dist


@partial(jax.jit, static_argnums=(2,))
def mi_knn(x, y, k: int = 3) -> jnp.array:
    """Mutual information using the KSG estimator.

    First algorithm proposed in Kraskov et al., Estimating mutual information,
    Phy rev, 2004.

    Parameters
    ----------
    x, y : array_like
        Arrays to consider for computing the Mutual Information. The two input
        variables x and y should have a shape of (n_features_x, n_samples) and
        (n_features_y, n_samples)
    k : int
        Number of nearest neighbors to consider for the KSG estimator.

    Returns
    -------
    mi : float
        Floating value describing the mutual-information between x and y.
    """
    # n_samples
    n = float(x.shape[1])
    # for each xi and yi, get the distance to neighbors
    eucl_xi = jnp.sqrt(_cdist(x, x))
    eucl_yi = jnp.sqrt(_cdist(y, y))

    # distance in space (XxY) is the maximum distance.
    max_dist_xy = jnp.maximum(eucl_xi, eucl_yi)
    # indices to the closest points in the (XxY) space.
    closest_points = jnp.argsort(max_dist_xy, axis=-1)
    # the kth neighbour is at index k (ignoring the point itself)
    # distance to the k-th neighbor for each point
    k_neighbours = closest_points[:, k]
    dist_k = max_dist_xy[jnp.arange(len(k_neighbours)), k_neighbours]
    # don't include the `i`th point itself in nx and ny
    nx = (eucl_xi < dist_k[:, None]).sum(axis=1) - 1
    ny = (eucl_yi < dist_k[:, None]).sum(axis=1) - 1

    psi_mean = jnp.sum((psi(nx + 1) + psi(ny + 1)) / n)

    mi = psi(k) - psi_mean + psi(n)
    return mi / jnp.log(2)
