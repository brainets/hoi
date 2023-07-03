import itertools
from math import comb
import itertools
from functools import partial
import logging

import numpy as np

import jax
import jax.numpy as jnp

from hoi.core.entropies import get_entropy, prepare_for_entropy
from hoi.core.combinatory import combinations

logger = logging.getLogger("frites")


###############################################################################
###############################################################################
#                                 ENTROPY
###############################################################################
###############################################################################

@partial(jax.jit, static_argnums=(2,))
def compute_entropies(x, idx, entropy=None):
    return x, entropy(x[:, idx, :])


###############################################################################
###############################################################################
#                                ITERATOR
#
# iterator to generate the combination of entropies i.e. :
# (H1, H2, H3); (H12, H13, H23); (H123)
###############################################################################
###############################################################################


def micomb(n, maxsize):
    combs, order = [], []
    for k in range(n):
        for i in itertools.combinations(range(n), k + 1):
            combs += [jnp.asarray(list(i) + [-1] * (maxsize - k - 1))]
            order += [len(i)]
    return jnp.asarray(combs), jnp.asarray(order).reshape(-1, 1)


###############################################################################
###############################################################################
#                            MUTUAL INFORMATION
###############################################################################
###############################################################################


@jax.jit
def find_entropy_index(inputs, mv):
    h_idx, comb = inputs

    # define indices of the multiplet
    mvm = jnp.where(mv != -1, comb[mv], -1).reshape(1, -1)

    # find this multiplet in the large list
    idx = jnp.where((h_idx == mvm).all(1), jnp.arange(h_idx.shape[0]), 0).sum()

    return (h_idx, comb), idx


@jax.jit
def sum_entropies(inputs, comb):
    h_x, h_idx, mvmidx, order = inputs

    # find order and indices
    _, idx = jax.lax.scan(
        find_entropy_index, (h_idx, comb), mvmidx
    )

    _h_x = ((-1.) ** (order - 1) * h_x[idx, :]).sum()

    return (h_x, h_idx, mvmidx, order), _h_x


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

    # prepare the data for computation
    data, kwargs = prepare_for_entropy(
        data, method, None, **kwargs
    )

    # ______________________________ ENTROPIES ________________________________

    # get the function to compute entropy and vmap it one for 3D inputs
    entropy = jax.jit(jax.vmap(get_entropy(method=method, **kwargs)))
    get_ent = jax.jit(partial(compute_entropies, entropy=entropy))

    logger.info(f"Compute entropies")

    h_x, h_idx = [], []
    for msize in range(1, maxsize + 1):
        logger.info(f"    Order={msize}")

        # compute all of the entropies at that order
        _h_idx = combinations(n_features, msize)
        _, _h_x = jax.lax.scan(get_ent, data, _h_idx)

        # add -1 for missing indices
        _h_idx = jnp.concatenate((
            jnp.asarray(_h_idx), jnp.full((len(_h_idx), maxsize - msize), -1)
        ), axis=1)

        # store entopies and indices associated to entropies
        h_x.append(_h_x)
        h_idx.append(_h_idx)

    h_x = jnp.concatenate(h_x, axis=0)
    h_idx = jnp.concatenate(h_idx, axis=0)

    # _______________________________ INFOTOPO ________________________________

    logger.info(f"Compute infotopo")

    hoi = []
    for msize in range(minsize, maxsize + 1):

        # combinations over spatial dimension
        combs = combinations(n_features, msize)

        # get formula of entropy summation
        mvmidx, order = micomb(msize, maxsize)

        # sum entropies
        _, _hoi = jax.lax.scan(
            sum_entropies, (h_x, h_idx, mvmidx, order), combs
        )

        hoi.append(_hoi)

    hoi = np.concatenate(hoi, axis=0)

    return hoi


if __name__ == '__main__':
    from math import comb as ccomb
    import matplotlib.pyplot as plt
    from frites import set_mpl_style
    import seaborn as sns
    import time as tst
    from hoi.utils import landscape, digitize
    from matplotlib.colors import LogNorm

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

    # x = digitize(x, 8, axis=0)
    oinfo = infotopo(x[..., 100], minsize=1, maxsize=None, method=method)
    0/0
    # plt.plot(oinfo)
    # plt.show()
    # 0/0
    # print(oinfo.shape)
    # print(oinfo.shape)
    # print(combs.shape)

    order = []
    for o in range(1, 4 + 1):
        order += [o] * ccomb(x.shape[1], o)
    print(oinfo.shape, len(order))

    lscp = landscape(oinfo.squeeze(), order, output='xarray', stat='count')
    lscp.plot(x='order', y='bins', cmap='jet', norm=LogNorm())
    plt.title(method, fontsize=24, fontweight='bold')
    plt.show()
    0/0

    vmin, vmax = np.percentile(oinfo, [1, 99])
    minmax = min(abs(vmin), abs(vmax))

    # plt.pcolormesh(oinfo, cmap='RdBu_r', vmin=-minmax, vmax=minmax)
    plt.pcolormesh(oinfo, cmap='RdBu_r')
    plt.colorbar()
    plt.show()
