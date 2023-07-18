from functools import partial

import jax
import jax.numpy as jnp


###############################################################################
###############################################################################
#                            ENTROPY BASED MI
###############################################################################
###############################################################################


@partial(jax.jit, static_argnums=(2,))
def mi_entr_comb(inputs, comb, entropy=None):
    """Entropy-based mutual information of a combination.

    This function can be used to scan mutual-information computations over
    set of multiplets.

    .. math::

        I_{x_{c}y} = H(x_{comb}) + H(y) - H(x_{comb}, y)

    Parameters
    ----------
    inputs : tuple
        Tuple that contains the x and y variable
    comb : jnp.array
        Jax array that contains the multiplet to select.

    Returns
    -------
    inputs : tuple
        Original inputs
    mi : jnp.array
        The mutual information
    """
    # unwrap arguments
    x, y = inputs

    # select combination
    xc = x[:, comb, :]

    # compute entropies
    h_x = entropy(xc)
    h_y = entropy(y)
    h_xy = entropy(jnp.concatenate((xc, y), axis=1))

    # compute mutual information
    mi = h_x + h_y - h_xy

    return inputs, mi
