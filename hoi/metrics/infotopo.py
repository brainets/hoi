import itertools
from math import comb
import itertools
from functools import partial
import logging

import numpy as np

import jax
import jax.numpy as jnp

from hoi.core.entropies import get_entropy, copnorm_nd
from hoi.core.combinatory import combinations
from hoi.core.entropies import entropy_gcmi

logger = logging.getLogger("frites")


def mvmi_combinations(n):
    combinations = []
    def_range = np.arange(n)
    for k in range(n):
        combinations += [np.array(m) for m in itertools.combinations(
            def_range, k + 1)]
    return combinations


@partial(jax.jit, static_argnums=(2,))
def compute_entropies(x, idx, entropy=None):
    return x, entropy(x[:, idx, :])


@partial(jax.jit)
def find_entropy_index(comb_idxn, target):
    n_comb, n_feat = comb_idxn.shape
    target = target.reshape(1, -1)
    at = jnp.where((comb_idxn == target).all(1), jnp.arange(n_comb), 0).sum()
    return comb_idxn, at


def infotopo(
        data, minsize=2, maxsize=None, method='gcmi', **kwargs
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

    # extract variables
    n_samples, n_features, n_variables = data.shape

    # get the maximum size of the multiplets investigated
    if not isinstance(maxsize, int):
        maxsize = n_features
    maxsize = max(1, maxsize)
    assert maxsize >= minsize

    logger.info(
        f"Compute the HOI (min={minsize}; max={maxsize})"
    )

    # ____________________________ PREPROCESSING ______________________________

    # method specific preprocessing
    if method == 'gcmi':
        logger.info('    copnorm data')
        data = copnorm_nd(data, axis=0)
        data = data - data.mean(axis=0, keepdims=True)
        kwargs['demean'] = False
    elif method == 'binning':
        if data.dtype != int:
            raise ValueError(
                "data dtype should be integer. Check that you discretized your"
                " data. If so, use `data.astype(int)`"
            )
        if 'n_bins' not in kwargs.keys():
            kwargs['n_bins'] = len(np.unique(data))
            logger.info(f"    {kwargs['n_bins']} bins detected from the data")
        n_bins = kwargs['n_bins']
        if (data.min() != 0) or (data.max() != n_bins - 1):
            raise ValueError(f"Values in data should be comprised between "
                             f"[0, n_bins={n_bins}]")
        kwargs['n_bins'] = n_bins

    # make the data (n_variables, n_features, n_trials)
    data = jnp.asarray(data.transpose(2, 1, 0))

    # ______________________________ ENTROPIES ________________________________

    # get the function to compute entropy and vmap it one for 3D inputs
    entropy = jax.vmap(get_entropy(method=method, **kwargs))
    get_ent = jax.jit(partial(compute_entropies, entropy=entropy))

    logger.info(f"Compute entropies")

    h_x, h_idx = [], []
    for msize in range(1, maxsize + 1):
        logger.info(f"    Order={msize}")

        # compute all of the entropies at that order
        _h_idx = combinations(n_features, msize)
        _, _h_x = jax.lax.scan(get_ent, data, _h_idx)

        # store entopies and indices associated to entropies
        h_x.append(np.asarray(_h_x))
        h_idx.append(_h_idx)

    # _______________________________ INFOTOPO ________________________________

    logger.info(f"Compute infotopo")

    hoi = []
    for msize in range(minsize, maxsize + 1):
        logger.info(f"    Order={msize}")

        # combinations over spatial dimension
        combs = combinations(n_features, msize)

        # get formula of entropy summation
        mvmidx = mvmi_combinations(msize)

        _hoi = np.zeros((len(combs), n_features))
        for m in mvmidx:
            # build indices specific to the multiplets
            _idx = combs[:, m]

            # number of features in this entropy
            m_order = len(m)
            h_x_m, h_idx_mi = h_x[m_order - 1], h_idx[m_order - 1]

            # find _idx inside h_idx_mi
            _, indices = jax.lax.scan(find_entropy_index, h_idx_mi, _idx)
            _hoi += ((-1) ** (m_order - 1)) * h_x_m[indices, :]

        hoi.append(_hoi)

    hoi = np.concatenate(hoi, axis=0)

    return hoi


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from frites import set_mpl_style
    import seaborn as sns
    import time as tst
    from hoi.utils import landscape, digitize

    set_mpl_style()

    np.random.seed(0)

    ###########################################################################
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


    oinfo = infotopo(x[..., 100], minsize=3, maxsize=None, method='gcmi')
    # plt.plot(oinfo)
    # plt.show()
    0/0
    print(oinfo.shape)
    # print(oinfo.shape)
    # print(combs.shape)

    # lscp = landscape(oinfo.squeeze(), combs, output='xarray')
    # lscp.plot(x='order', y='bins')
    # plt.show()
    # 0/0

    vmin, vmax = np.percentile(oinfo, [1, 99])
    minmax = min(abs(vmin), abs(vmax))

    # plt.pcolormesh(oinfo, cmap='RdBu_r', vmin=-minmax, vmax=minmax)
    plt.pcolormesh(oinfo, cmap='RdBu_r')
    plt.colorbar()
    plt.show()
