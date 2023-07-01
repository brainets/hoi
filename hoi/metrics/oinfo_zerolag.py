from math import comb
import itertools
from functools import partial
import logging

import numpy as np

import jax
import jax.numpy as jnp

from hoi.core.combinatory import combinations
from hoi.core.entropies import get_entropy, prepare_for_entropy
from hoi.core.oinfo import oinfo_scan

logger = logging.getLogger("frites")


def oinfo_zerolag(
        data, y=None, minsize=2, maxsize=None, method='gcmi', **kwargs
    ):
    """Dynamic, possibly task-related oinfo.

    Parameters
    ----------
    data : array_like
        Standard NumPy arrays of shape (n_samples, n_features) or
        (n_samples, n_features, n_variables)
    y : array_like
        The feature of shape (n_trials,) for estimating task-related O-info.
    minsize, maxsize : int | 2, None
        Minimum and maximum size of the multiplets
    method : {'gcmi', 'binning', 'knn'}
        Name of the method to compute entropy. Use either :

            * 'gcmi': gaussian copula entropy [default]
            * 'binning': binning-based estimator of entropy. Note that to use
              this estimator, the data have be to discretized
            * 'knn': k-nearest neighbor estimator
    kwargs : dict | {}
        Additional arguments are sent to each entropy function

    Returns
    -------
    oinfo : array_like
        The O-info array of shape (n_multiplets, n_variables) where positive
        values reflect redundant dominated interactions and negative values
        stand for synergistic dominated interactions.
    """
    # ________________________________ INPUTS _________________________________
    # force data to be 3d
    assert data.ndim >= 2
    if data.ndim == 2:
        data = data[..., np.newaxis]

    # inputs conversion
    is_task_related = isinstance(y, (list, np.ndarray, tuple))

    # extract variables
    n_samples, n_features, n_variables = data.shape

    # get the maximum size of the multiplets investigated
    if not isinstance(maxsize, int):
        maxsize = n_features
    maxsize = max(1, maxsize)
    assert maxsize >= minsize

    logger.info(
        f"Compute the {'task-related ' * is_task_related} HOI "
        f"(min={minsize}; max={maxsize})"
    )

    # ____________________________ PREPROCESSING ______________________________

    # prepare the data for computation
    data, kwargs = prepare_for_entropy(
        data, method, y, **kwargs
    )

    # ________________________________ O-INFO _________________________________

    # get the function to compute entropy and vmap it twice for 4D inputs
    entropy = jax.jit(jax.vmap(jax.vmap(
        get_entropy(method=method, **kwargs)
    )))

    # use it to compute oinfo
    oinfo_mmult = jax.jit(partial(oinfo_scan, entropy=entropy))


    oinfo = []
    for msize in range(minsize, maxsize + 1):
        logger.info(f"    Multiplets of size {msize}")
        combs = combinations(n_features, msize, task_related=is_task_related)

        _, _oinfo = jax.lax.scan(oinfo_mmult, data, combs)
        oinfo.append(np.asarray(_oinfo))

    oinfo = np.concatenate(oinfo, axis=0)
    return oinfo


if __name__ == '__main__':
    from math import comb as ccomb
    import matplotlib.pyplot as plt
    from frites import set_mpl_style
    import seaborn as sns
    import time as tst
    from matplotlib.colors import LogNorm

    from hoi.utils import landscape, digitize

    set_mpl_style()

    np.random.seed(0)

    ###########################################################################
    method = 'gcmi'
    n_trials = 600
    n_roi = 5
    n_times = 50

    redundancy = [
        (2, 3, 4)
    ]
    synergy = [
        (0, 1, 2)
    ]
    ###########################################################################

    def set_redundancy(x, redundancy, sl, win, trials):
        for m in redundancy:
            x[:, m, sl] += .8 * trials.reshape(-1, 1, 1) * win
        return x

    def set_synergy(x, synergy, sl, win, trials):
        for m in synergy:
            blocks = np.array_split(np.arange(n_trials), len(m))
            for n_b, b in enumerate(blocks):
                x[b, m[n_b], sl] += trials[b].reshape(-1, 1) * win[0, ...]
        return x


    # generate the data
    x = np.random.rand(n_trials, n_roi, n_times)
    roi = np.array([f"r{r}" for r in range(n_roi)])[::-1]
    trials = np.random.rand(n_trials)
    times = np.arange(n_times)
    win = np.hanning(10).reshape(1, 1, -1)

    # introduce (redundant, synergistic) information
    x = set_redundancy(x, redundancy, slice(20, 30), win, trials)
    x = set_synergy(x, synergy, slice(30, 40), win, trials)

    start_time = tst.time()

    x = np.load('/home/etienne/Downloads/data_time_evolution', allow_pickle=True)
    # for nt in range(x.shape[-1]):
    #     x[:, :, nt] = digitize(x[:, :, nt], 8)
    # x = x.astype(int)
    x = x[..., 100]

    oinfo = oinfo_zerolag(x, minsize=2, maxsize=7, method=method)
    print(oinfo)

    order = []
    for o in range(2, 7 + 1):
        order += [o] * ccomb(x.shape[1], o)

    lscp = landscape(oinfo.squeeze(), order, output='xarray')
    lscp.plot(x='order', y='bins', cmap='turbo', norm=LogNorm())
    plt.title(method, fontsize=24, fontweight='bold')
    plt.show()
    0/0

    vmin, vmax = np.percentile(oinfo, [1, 99])
    minmax = min(abs(vmin), abs(vmax))

    # plt.pcolormesh(oinfo, cmap='RdBu_r', vmin=-minmax, vmax=minmax)
    plt.pcolormesh(oinfo, cmap='RdBu_r')
    plt.colorbar()
    plt.show()
