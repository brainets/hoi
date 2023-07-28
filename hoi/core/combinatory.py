import jax.numpy as jnp
import numpy as np
import itertools
from math import comb as ccomb


def _combinations(n, k, order):
    for c in itertools.combinations(range(n), k):
        # convert to list
        c = list(c)

        # deal with order
        if order:
            c = len(c)

        yield c


def combinations(n, k, astype="iterator", order=False):
    """Get combinations.

    Parameters
    ----------
    n : int
        Represents the total number of elements in the set
    k : int
        Represents the size of the combinations to be generated

    Returns
    -------
    combs : jnp.array
        An array of shape (n_combinations, k) representing all possible
        combinations of k elements.
    """

    iterator = _combinations(n, k, order)

    assert astype in ['iterator', 'jax', 'numpy']
    if astype == "iterator":
        return iterator
    elif astype in ['jax', 'numpy']:
        n_mults = ccomb(n, k)
        n_cols = 1 if order else k

        combs = np.zeros((n_mults, n_cols), dtype=int)
        for n_c, c in enumerate(iterator):
            combs[n_c, :] = c

        if astype == 'jax':
            combs = jnp.asarray(combs)

        return combs


if __name__ == '__main__':
    print(type(combinations(10, 5, astype='jax', order=False)))

    # print(np.array(list(itertools.combinations(np.arange(10), 3))).shape)