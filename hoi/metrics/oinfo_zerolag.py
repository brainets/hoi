from math import comb
import jax
import jax.numpy as jnp
import numpy as np
import logging
import itertools

logger = logging.getLogger("frites")

from hoi.core.it import ctransform, copnorm_1d, copnorm_nd, oinfo_mmult, ent_vector_g
from hoi.core.combinatory import combinations

# vmapping the ent_vector_g function
ent_vector_vmap = jax.vmap(ent_vector_g)


def oinfo_zerolag(data, y=None, minsize=3, maxsize=5):
    """Dynamic, possibly task-related oinfo.

    Parameters
    ----------
    data : array_like
            Standard NumPy arrays of shape (n_samples, n_features, n_variables)
    y : array_like
        The feature of shape (n_trials,) for estimating task-related O-info.
    minsize, maxsize : int | 3, 5
        Minimum and maximum size of the multiplets

    Returns
    -------
    oinfo : array_like
        The O-info array of shape (n_multiplets, n_variables) where positive values
        reflect redundant dominated interactions and negative values stand for
        synergistic dominated interactions.
    """
    # ________________________________ INPUTS _________________________________
    # inputs conversion
    is_task_related = isinstance(y, (str, list, np.ndarray, tuple))

    # extract variables
    # x = data
    n_samples, n_features, n_variables = data.shape

    # get the maximum size of the multiplets investigated
    if not isinstance(maxsize, int):
        maxsize = n_features
    maxsize = max(1, maxsize)
    assert maxsize > minsize

    # get the number of multiplets
    n_mults = 0
    for msize in range(minsize, maxsize + 1):
        n_mults += comb(n_features, msize)

    logger.info(
        f"Compute the {'task-related ' * is_task_related} HOI "
        f"(min={minsize}; max={maxsize})"
    )

    # ________________________________ O-INFO _________________________________

    logger.info("    Copnorm the data")

    # for task-related, add behavior along spatial dimension
    if is_task_related:
        y = np.tile(y.reshape(-1, 1, 1), (1, 1, n_variables))
        data = np.concatenate((data, data), axis=1)
        # n_features += 1

    # copnorm and demean the data
    data = copnorm_nd(data.copy(), axis=0)
    data = data - data.mean(axis=0, keepdims=True)

    # make the data (n_variables, n_features, n_trials)
    data = jnp.asarray(data.transpose(2, 1, 0))

    oinfo = []
    for msize in range(minsize, maxsize + 1):
        logger.info(f"    Multiplets of size {msize}")
        combs = combinations(n_features, msize, task_related=is_task_related)

        _, _oinfo = jax.lax.scan(oinfo_mmult, data, combs)
        oinfo.append(np.asarray(_oinfo))

    oinfo = np.concatenate(oinfo, axis=0)
    return oinfo
