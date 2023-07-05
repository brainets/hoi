import itertools
from math import comb
import itertools
from functools import partial
import logging

import numpy as np

import jax
import jax.numpy as jnp

from hoi.metrics.base_hoi import HOIEstimator

logger = logging.getLogger("frites")


###############################################################################
###############################################################################
#                                ITERATOR
#
# iterator to generate the combination of entropies i.e. :
# (H1, H2, H3); (H12, H13, H23); (H123)
###############################################################################
###############################################################################


def _micomb(n, k):
    def_range = jnp.arange(n)
    return jnp.stack([np.array(m) for m in itertools.combinations(
        def_range, k + 1) if n - 1 in m], axis=0)

def micomb(n):
    return map(lambda k: _micomb(n, k), range(n))


###############################################################################
###############################################################################
#                                 ENTROPY
###############################################################################
###############################################################################

@partial(jax.jit, static_argnums=(2,))
def compute_entropies(x, idx, entropy=None):
    """Compute entropy for a specific multiplet.

    This function has to be wrapped with the entropy function.
    """
    return x, entropy(x[:, idx, :])


###############################################################################
###############################################################################
#                            MUTUAL INFORMATION
###############################################################################
###############################################################################


@jax.jit
def find_entropy_index(comb_idxn, target):
    n_comb, n_feat = comb_idxn.shape
    target = target.reshape(1, -1)
    at = jnp.where((comb_idxn == target).all(1), jnp.arange(n_comb), 0).sum()
    return comb_idxn, at


@jax.jit
def sum_entropies(inputs, m):
    combs, h_x_m, h_idx_m, sgn, _hoi = inputs

    # build indices specific to the multiplets
    _idx = combs[:, m]

    # find _idx inside h_idx_m
    indices = (
        _idx[..., jnp.newaxis] == h_idx_m.T[jnp.newaxis, ...]).sum(1).argmax(1)


    _hoi += sgn * h_x_m[indices, :]

    return (combs, h_x_m, h_idx_m, sgn, _hoi), None



class InfoTopo(HOIEstimator):

    """Dynamic, possibly task-related Topological Information."""

    def __init__(self):
        HOIEstimator.__init__(self)

    def fit(self, data, y=None, maxsize=None, method='gcmi',
            **kwargs):
        """Compute Topological Information.

        Parameters
        ----------
        data : array_like
            Standard NumPy arrays of shape (n_samples, n_features) or
            (n_samples, n_features, n_variables)
        y : array_like
            The feature of shape (n_trials,) for estimating task-related O-info.
        maxsize : int | None
            Maximum size of the multiplets
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
        self._prepare_multiplets(1, maxsize, y=y)

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

        h_x, h_idx = [], []
        for msize in self:
            logger.info(f"    Order={msize}")

            # compute all of the entropies at that order
            _h_idx = self.get_combinations(msize)
            _, _h_x = jax.lax.scan(get_ent, data, _h_idx)

            # store entopies and indices associated to entropies
            h_x.append(_h_x)
            h_idx.append(_h_idx)

        # _____________________________ INFOTOPO ______________________________

        logger.info(f"Compute infotopo")

        hoi = []
        for msize in self:

            logger.info(f"    Order={msize}")

            # combinations over spatial dimension
            combs = self.get_combinations(msize)

            # order 1, just select entropies
            if msize == 1:
                np.testing.assert_array_equal(
                    h_idx[0].squeeze(), combs.squeeze())
                hoi.append(h_x[0])
                continue

            # find indices associated to cmi_{n-1}
            combs_prev = combs[:, 0:-1]
            if combs_prev.ndim == 1:
                combs_prev = combs_prev[:, jnp.newaxis]
            _, mi_prev_idx = jax.lax.scan(
                find_entropy_index, h_idx[msize - 2], combs_prev)
            np.testing.assert_array_equal(
                h_idx[msize - 2][mi_prev_idx, :], combs_prev)

            # initialize cmi_{n + 1} = f(cmi_{n})
            _hoi = hoi[-1][mi_prev_idx ,:]

            # terms for entropy summation from cmi_{n+1}, without cmi_{n}
            mvmidx = micomb(msize)


            for n_m, m in enumerate(mvmidx):
                # define order
                m_order = m.shape[1] - 1

                # order selection
                h_x_m, h_idx_m = h_x[m_order], h_idx[m_order]
                sgn = (-1.) ** m_order

                # scan over tuples
                (_, _, _, _, _hoi), _ = jax.lax.scan(
                    sum_entropies, (combs, h_x_m, h_idx_m, sgn, _hoi), m
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

    # x = digitize(x, 9, axis=0)
    model = InfoTopo()
    hoi = model.fit(
        x[..., 100], maxsize=None, method=method
    )
    0/0

    lscp = landscape(hoi.squeeze(), model.order, output='xarray')
    lscp.plot(x='order', y='bins', cmap='jet', norm=LogNorm())
    plt.axvline(model.undersampling, linestyle='--', color='k')
    plt.title(method, fontsize=24, fontweight='bold')
    plt.show()
