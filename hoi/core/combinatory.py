import jax.numpy as jnp
import numpy as np
import itertools


def combinations(n, k, task_related=False):
    """Get combinations.
    
    Parameters
    ----------
    n : int
        Represents the total number of elements in the set
    k : int
        Represents the size of the combinations to be generated
    task_related : bool, optional 
        If True, include an additional column in the returned array representing task-related behavior, by default False.

    Returns
    -------
    combs : jnp.array 
        An array of shape (n_combinations, k) representing all possible 
        combinations of k elements.
    """
    combs = np.array(list(itertools.combinations(np.arange(n), k)))

    # add behavior as a final columns
    if task_related:
        combs = np.c_[combs, np.full((combs.shape[0],), n)]

    return jnp.asarray(combs)
