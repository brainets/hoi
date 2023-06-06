from math import comb
import jax
import jax.numpy as jnp
import numpy as np
import logging

logger = logging.getLogger("frites")
from hoi.oinfo.conn_oinfo_jax import combinations
from hoi.oinfo.conn_oinfo_jax import oinfo_mmult

from hoi.core.it import ctransform, copnorm_1d, copnorm_nd


def oinfo_zerolag(
    data,
    y=None,
    variables=None,
    features=None,
    minsize=3,
    maxsize=5,
    sort=True,
    verbose=None,
):
    """Dynamic, possibly task-related oinfo.

    Parameters
    ----------
    data : array_like
            Standard NumPy arrays of shape (n_samples, n_features, n_variables)
    y : array_like
        The feature of shape (n_trials,) for estimating task-related O-info.
    features : array_like | None
        Array of region of interest name of shape (n_features,)
    variables : array_like | None
        Array of time points of shape (n_variables,)
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
    # kw_links = {"directed": False, "net": False}

    # extract variables
    # x, attrs = data.data, cfg["attrs"]
    y, features, variables = (
        data["y"].data,
        data["features"].data,
        data["variables"].data,
    )
    n_features = len(features)

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
        y = np.tile(y.reshape(-1, 1, 1), (1, 1, len(variables)))
        x = np.concatenate((x, y), axis=1)
        features = np.r_[features, ["beh"]]

    # copnorm and demean the data
    x = copnorm_nd(x.copy(), axis=0)
    x = x - x.mean(axis=0, keepdims=True)

    # make the data (n_variables, n_features, n_trials)
    x = jnp.asarray(x.transpose(2, 1, 0))

    oinfo, features_o = [], []
    for msize in range(minsize, maxsize + 1):
        # ----------------------------- MULTIPLETS ----------------------------
        logger.info(f"    Multiplets of size {msize}")
        combs, _features_o = combinations(
            n_features, msize, features, task_related=is_task_related, sort=sort
        )
        features_o += _features_o

        # ------------------------------- O-INFO ------------------------------
        _, _oinfo = jax.lax.scan(oinfo_mmult, x, combs)

        oinfo.append(np.asarray(_oinfo))
    oinfo = np.concatenate(oinfo, axis=0)

    return oinfo
