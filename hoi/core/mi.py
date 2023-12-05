from functools import partial

import jax
import jax.numpy as jnp

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
    # get the entropy unction
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


###############################################################################
###############################################################################
#                                 OTHERS
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
