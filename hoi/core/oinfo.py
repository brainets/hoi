from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(2,))
def oinfo_scan(
        x: jnp.array, comb: jnp.array, entropy=None
    ) -> (jnp.array, jnp.array):
    """Compute the O-information.

    Parameters
    ----------
    x : array_like
        Input data of shape (n_variables, n_features, n_trials)
    comb : array_like
        Combination to use (e.g. (0, 1, 2))
    entropy : callable | None
        Entropy function to use for the computation

    Returns
    -------
    oinfo : array_like
        O-information for the multiplet comb
    """
    # build indices
    msize = len(comb)
    ind = jnp.mgrid[0:msize, 0:msize].sum(0) % msize
    ind = ind[:, 1:]

    # multiplet selection
    x_mult = x[:, comb, :]
    nvars = x_mult.shape[-2]

    # compute the entropies
    h_n = entropy(x_mult[:, jnp.newaxis, ...])[:, 0]
    h_j = entropy(x_mult[..., jnp.newaxis, :])
    h_mj = entropy(x_mult[..., ind, :])

    o = (nvars - 2) * h_n + (h_j - h_mj).sum(1)

    return x, o
