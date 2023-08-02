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


def combinations(
    n, minsize, maxsize=None, astype="iterator", order=False, fill_value=-1
):
    """Get combinations.

    Parameters
    ----------
    n : int
        Represents the total number of elements in the set
    minsize : int
        Minimum size of the multiplets
    maxsize : int | None
        Maximum size of the multiplets. If None, minsize is used.
    astype : {'jax', 'numpy', 'iterator'}
        Specify the output type. Use either 'jax' get the data as a jax
        array [default], 'numpy' for NumPy array or 'iterator'.
    order : bool, optional
        If True, return the order of each multiplet. Default is False.

    Returns
    -------
    combs : jnp.array
        An array of shape (n_combinations, k) representing all possible
        combinations of k elements.
    """
    # ________________________________ ITERATOR _______________________________
    if not isinstance(maxsize, int):
        maxsize = minsize
    assert maxsize >= minsize
    iterators = []
    for msize in range(minsize, maxsize + 1):
        iterators.append(_combinations(n, msize, order))
    iterators = itertools.chain(*tuple(iterators))

    if astype == "iterator":
        return iterators

    # _________________________________ ARRAYS ________________________________
    if order:
        combs = np.asarray([c for c in iterators]).astype(int)
    else:
        # get the number of combinations
        n_mults = sum([ccomb(n, c) for c in range(minsize, maxsize + 1)])

        # prepare output
        combs = np.full((n_mults, maxsize), fill_value, dtype=int)

        # fill the array
        for n_c, c in enumerate(iterators):
            combs[n_c, 0:len(c)] = c

    # jax conversion (f required)
    if astype == "jax":
        combs = jnp.asarray(combs)

    return combs


if __name__ == "__main__":
    print(combinations(10, minsize=2, maxsize=None, astype="jax", order=False))

    # print(np.array(list(itertools.combinations(np.arange(10), 3))).shape)
