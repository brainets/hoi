from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import digamma as psi

from .entropies import get_entropy

###############################################################################
###############################################################################
#                                 SWITCHER
###############################################################################
###############################################################################


def get_mi(method="gcmi", **kwargs):
    """Get Mutual-Information function.

    Parameters
    ----------
    method : {'gcmi', 'binning', 'knn', 'kernel'}
        Name of the method to compute mutual-information.
    kwargs : dict | {}
        Additional arguments sent to the mutual-information function.

    Returns
    -------
    fcn : callable
        Function to compute mutual information on variables of shapes
        (n_features, n_samples)
    """
    if method == "knn":
        return partial(compute_mi_knn, **kwargs)
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
#     x, kwargs = prepare_for_entropy(x, method, **kwargs.copy())
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
#                         KNN MUTUAL INFORMATION
###############################################################################
###############################################################################


@partial(jax.jit, static_argnums=(2,))
def n_neighbours(xy, idx, k=1):
    """Return number of neighbours for each point based on kth neighbour."""
    xi, x = xy[0][:, [idx]], xy[0]
    yi, y = xy[1][:, [idx]], xy[1]

    # compute euclidian distance from xi to all points in x (same y)
    eucl_xi = jnp.sqrt(jnp.sum((xi - x) ** 2, axis=0))
    eucl_yi = jnp.sqrt(jnp.sum((yi - y) ** 2, axis=0))

    # distance in space (XxY) is the maximum distance.
    max_dist_xy = jnp.maximum(eucl_xi, eucl_yi)
    # indices to the closest points in the (XxY) space.
    closest_points = jnp.argsort(max_dist_xy)
    # the kth neighbour is at index k (ignoring the point itself)
    # distance to the k-th neighbor for each point
    dist_k = max_dist_xy[closest_points[k]]
    # don't include the `i`th point itself in nx and ny
    nx = (eucl_xi < dist_k).sum() - 1
    ny = (eucl_yi < dist_k).sum() - 1

    return xy, (nx, ny)


@partial(jax.jit, static_argnums=(2,))
def compute_mi_knn(x, y, k: int = 1) -> jnp.array:
    """Mutual information using the KSG estimator.

    First algorithm proposed in Kraskov et al., Estimating mutual information,
    Phy rev, 2004.

    Parameters
    ----------
    x, y : array_like
        Input data of shape (n_features, n_samples).
    k : int
        Number of nearest neighbors to consider for the KSG estimator.

    Returns
    -------
    mi : float
        Floating value describing the mutual-information between x and y.
    """
    # n_samples
    n = float(x.shape[1])

    _n_neighbours = partial(n_neighbours, k=k)
    # get number of neighbors for each point in XxY space
    _, n_neighbors = jax.lax.scan(
        _n_neighbours, (x, y), jnp.arange(int(n)).astype(int)
    )
    nx = n_neighbors[0]
    ny = n_neighbors[1]

    psi_mean = jnp.sum((psi(nx + 1) + psi(ny + 1)) / n)

    mi = psi(k) - psi_mean + psi(n)
    return mi / jnp.log(2)
