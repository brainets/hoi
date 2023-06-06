import jax.numpy as jnp
import numpy as np
import itertools


def combinations(n, k, feature, task_related=False, sort=True):
    """Get combinations.
    
    Parameters
    ----------
    n : int
        Represents the total number of elements in the set
    k : int
        Represents the size of the combinations to be generated
    feature : array_like
        Contains the names of the elements in the set
    task_related : bool, optional 
        If True, include an additional column in the returned array representing task-related behavior, by default False.
    sort : a boolean flag (default is True) indicating whether to sort the 
    combinations lexicographically

    Returns
    -------
    combs : jnp.array 
        An array of shape (n_combinations, k) representing all possible 
        combinations of k elements.
    feature_st : list  
        A list of strings representing the names of each combination. If sort is True, 
        the names will be sorted alphabetically. If task_related is True, 
        the last column of each combination will represent the behavior.
    """
    combs = np.array(list(itertools.combinations(np.arange(n), k)))

    # add behavior as a final columns
    if task_related:
        combs = np.c_[combs, np.full((combs.shape[0],), n)]

    # build feature names
    if not sort:
        feature_st = ["-".join(r) for r in feature[combs].tolist()]
    else:
        feature_st = ["-".join(r) for r in np.sort(feature[combs]).tolist()]

    return jnp.asarray(combs), feature_st
