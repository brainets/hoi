import jax.numpy as jnp
import numpy as np
import itertools


def combinations(n, k, as_iterator=True, as_jax=True, order=False):
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

    def _combinations(n, k):
        for c in itertools.combinations(np.arange(n), k):
            # convert to list
            c = list(c)

            # deal with order
            if order:
                c = len(c)

            yield c


    if as_iterator:
        return _combinations(n, k)
    else:
        combs = [c for c in _combinations(n, k)]
        return jnp.asarray(combs) if as_jax else combs

if __name__ == '__main__':
    print(combinations(10, 5, as_iterator=False, order=True).shape)
    # print(np.array(list(itertools.combinations(np.arange(10), 3))).shape)