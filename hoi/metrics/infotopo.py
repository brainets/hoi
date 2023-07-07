import itertools
from math import comb
import itertools
from functools import partial
import logging

from tqdm import tqdm

import numpy as np

import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm

from hoi.metrics.base_hoi import HOIEstimator

logger = logging.getLogger("hoi")


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
def find_indices(inputs, c):
    combs, keep = inputs

    keep = jnp.add(keep, (combs == c).any(1).astype(int))

    return (combs, keep), None


@jax.jit
def compute_mi(inputs, iterators):
    combs, h, order = inputs
    _, comb = iterators

    # scanning over indices
    # is_inside = jnp.zeros((combs.shape[0],), dtype=int)
    # (_, is_inside), _ = jax.lax.scan(find_indices, (combs, is_inside), comb)

    # tensor implementation
    is_inside = (combs == comb[jnp.newaxis, ...]).any(1).sum(1)


    return inputs, jnp.sum(h, where=(is_inside == order).reshape(-1, 1))



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
                * 'binning': binning-based estimator of entropy. Note that to
                  use this estimator, the data have be to discretized
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

        logger.info("    Compute entropies")

        # compute infotopo
        h_x, h_idx, order = [], [], []
        for msize in self:
            # combinations of features
            combs = self.get_combinations(msize)

            # compute all of the entropies at that order
            _, _h_x = jax.lax.scan(get_ent, data, combs)

            # appen -1 to the combinations
            combs = jnp.concatenate(
                (combs, jnp.full((combs.shape[0], self.maxsize - msize), -1)),
                axis=1
            )

            # store entopies and indices associated to entropies
            h_x.append(_h_x)
            h_idx.append(combs)
            order.append([msize] * _h_x.shape[0])

        h_x = jnp.concatenate(h_x, axis=0)
        h_idx = jnp.concatenate(h_idx, axis=0)
        order = jnp.asarray(np.concatenate(order, axis=0))
        n_mult = h_x.shape[0]

        # ________________________ MUTUAL-INFORMATION _________________________

        # compute order and multiply entropies
        h_x_sgn = jnp.multiply(((-1.) ** (order.reshape(-1, 1) - 1)), h_x)
        h_idx_2 = jnp.where(h_idx == -1, -2, h_idx)

        logger.info("    Compute mutual information")

        pbar = scan_tqdm(n_mult, message='Mutual information')

        _, hoi = jax.lax.scan(
            pbar(compute_mi), (h_idx[..., jnp.newaxis], h_x_sgn, order),
            (jnp.arange(n_mult), h_idx_2)
        )

        # self._h = np.asarray(h_x)
        # self._h_idx = np.asarray(h_idx)

        return np.asarray(hoi)

    # @property
    # def entropies(self):
    #     """Computed entropies."""
    #     return np.concatenate([np.asarray(k) for k in self._h], axis=0)

    # @property
    # def entropies_indices(self):
    #     """Get multiplets associated to entropies."""
    #     mult = []
    #     for c in self._h_idx:
    #         mult += np.asarray(c).tolist()
    #     return mult



if __name__ == '__main__':
    from math import comb as ccomb
    import matplotlib.pyplot as plt
    from frites import set_mpl_style
    import seaborn as sns
    from hoi.utils import landscape, digitize
    from matplotlib.colors import LogNorm

    set_mpl_style()

    ###########################################################################
    method = 'binning'
    ###########################################################################


    x = np.load('/home/etienne/Downloads/data_200_trials', allow_pickle=True)

    logger.setLevel('INFO')
    model = InfoTopo()
    x = digitize(x, 9, axis=0)
    hoi = model.fit(
        x[..., 100], maxsize=None, method=method
    )

    lscp = landscape(hoi.squeeze(), model.order, output='xarray')
    lscp.plot(x='order', y='bins', cmap='jet', norm=LogNorm())
    plt.axvline(model.undersampling, linestyle='--', color='k')
    plt.title(method, fontsize=24, fontweight='bold')
    plt.show()
