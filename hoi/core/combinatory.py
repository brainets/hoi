import jax.numpy as jnp
import numpy as np
import itertools


def combinations(n, k, task_related=False, as_iterator=True, as_jax=True,
                 order=False):
    """Get combinations.

    Parameters
    ----------
    n : int
        Represents the total number of elements in the set
    k : int
        Represents the size of the combinations to be generated
    task_related : bool, optional
        If True, include an additional column in the returned array
        representing task-related behavior, by default False.

    Returns
    -------
    combs : jnp.array
        An array of shape (n_combinations, k) representing all possible
        combinations of k elements.
    """

    def _combinations(n, k):
        for c in itertools.combinations(np.arange(n), k):
            # if task related, force adding last column
            if task_related:
                c = list(c) + [n]
            else:
                c = list(c)

            # deal with order
            if order:
                if task_related:
                    c = len(c) - 1
                else:
                    c = len(c)

            yield c


    if as_iterator:
        return _combinations(n, k)
    else:
        combs = [c for c in _combinations(n, k)]
        return jnp.asarray(combs) if as_jax else combs

if __name__ == '__main__':
    print(combinations(10, 5, task_related=False, as_iterator=False).shape)
    # print(np.array(list(itertools.combinations(np.arange(10), 3))).shape)