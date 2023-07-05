import itertools
from math import comb
import itertools
from functools import partial
import logging

import numpy as np

import jax
import jax.numpy as jnp

from hoi.metrics.est_hoi import HOIEstimator

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
def find_entropy_index(all_comb, to_find):
    n_comb, n_feat = all_comb.shape
    to_find = to_find.reshape(1, -1)
    at = jnp.where((all_comb == to_find).all(1), jnp.arange(n_comb), 0).sum()
    return all_comb, at


@jax.jit
def sum_entropies(inputs, idxorder):
    h_x, h_idx, combs, _hoi = inputs
    mvmidx, order = idxorder

    # get the number of -1 to pad
    n_to_pad = jnp.where(mvmidx == -1, 1, 0).sum()

    # build indices specific to the multiplets
    _idx = combs[:, mvmidx]

    # pad indices with -1
    _idx = jnp.where(mvmidx.reshape(1, -1) == -1, -1, _idx)

    # find _idx inside all of the multiplets
    _, indices = jax.lax.scan(
        find_entropy_index, h_idx, _idx
    )

    # accumulate entropy on carry
    _hoi += ((-1) ** (order - 1)) * h_x[indices, :]

    return (h_x, h_idx, combs, _hoi), _



class InfoTopo(HOIEstimator):

    """Dynamic, possibly task-related Topological Information."""

    def __init__(self):
        HOIEstimator.__init__(self)

    def fit(self, data, y=None, minsize=1, maxsize=None, method='gcmi',
            **kwargs):
        """Compute Topological Information.

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
        # ______________________________ INPUTS _______________________________

        # data checking
        data = self._prepare_data(data)

        # check multiplets
        self._prepare_multiplets(minsize, maxsize, y=y)

        # check entropy function
        data, entropy = self._prepare_for_entropy(data, method, y=y, **kwargs)

        logger.info(
            f"Compute the info topo (min={self.minsize}; max={self.maxsize})"
        )

        # ____________________________ ENTROPIES ______________________________

        # get the function to compute entropy and vmap it one for 3D inputs
        get_ent = jax.jit(partial(
            compute_entropies, entropy=jax.vmap(entropy)
        ))

        logger.info(f"Compute entropies")

        h_x, h_idx, hoi = [], [], []
        for msize in range(1, self.maxsize + 1):
            logger.info(f"    Order={msize}")

            # compute all of the entropies at that order
            _h_idx = self.get_combinations(msize)
            _, _h_x = jax.lax.scan(get_ent, data, _h_idx)

            # add -1 for missing indices
            _h_idx = jnp.concatenate((
                jnp.asarray(_h_idx), jnp.full((len(_h_idx), self.maxsize - msize), -1)
            ), axis=1)

            # concatenate everything
            if isinstance(h_x, list):
                h_x = _h_x
            else:
                h_x = jnp.concatenate((h_x, _h_x), axis=0)

            if isinstance(h_idx, list):
                h_idx = _h_idx
            else:
                h_idx = jnp.concatenate((h_idx, _h_idx))

            # combinations over spatial dimension
            combs = self.get_combinations(msize)

            # get formula of entropy summation
            mvmidx, order = micomb(msize, self.maxsize)

            logger.info(f"    Order={msize}")

            # sum entropies
            _hoi = np.zeros((len(combs), self.n_variables))
            (_, _, _, _hoi), _ = jax.lax.scan(
                sum_entropies, (h_x, h_idx, combs, _hoi), (mvmidx, order)
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
    ###########################################################################


    x = np.load('/home/etienne/Downloads/data_200_trials', allow_pickle=True)

    # x = digitize(x, 8, axis=0)
    model = InfoTopo()
    hoi = model.fit(
        x[..., 100], minsize=3, maxsize=None, method=method
    )

    lscp = landscape(hoi.squeeze(), model.order, output='xarray')
    lscp.plot(x='order', y='bins', cmap='jet', norm=LogNorm())
    plt.axvline(model.undersampling, linestyle='--', color='k')
    plt.title(method, fontsize=24, fontweight='bold')
    plt.show()
