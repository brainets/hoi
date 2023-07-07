from math import comb
import itertools
from functools import partial
import logging

import numpy as np

import jax
import jax.numpy as jnp

from hoi.metrics.base_hoi import HOIEstimator

logger = logging.getLogger("hoi")


@partial(jax.jit, static_argnums=(2,))
def oinfo_scan(
        x: jnp.array, comb: jnp.array, entropy=None
    ) -> (jnp.array, jnp.array):
    """Compute the O-information.

    Parameters
    ----------
    x : array_like
        Input data of shape (n_variables, n_features, n_trials)
    comb : array_like
        Combination to use (e.g. (0, 1, 2))
    entropy : callable | None
        Entropy function to use for the computation

    Returns
    -------
    oinfo : array_like
        O-information for the multiplet comb
    """
    # build indices
    msize = len(comb)
    ind = jnp.mgrid[0:msize, 0:msize].sum(0) % msize
    ind = ind[:, 1:]

    # multiplet selection
    x_mult = x[:, comb, :]
    nvars = x_mult.shape[-2]

    # compute the entropies
    h_n = entropy(x_mult[:, jnp.newaxis, ...])[:, 0]
    h_j = entropy(x_mult[..., jnp.newaxis, :])
    h_mj = entropy(x_mult[..., ind, :])

    o = (nvars - 2) * h_n + (h_j - h_mj).sum(1)

    return x, o



class OinfoZeroLag(HOIEstimator):

    """Dynamic, possibly task-related O-info."""

    def __init__(self):
        HOIEstimator.__init__(self)

    def fit(self, data, y=None, minsize=2, maxsize=None, method='gcmi',
            **kwargs):
        """Compute the O-information.

        Parameters
        ----------
        data : array_like
            Standard NumPy arrays of shape (n_samples, n_features) or
            (n_samples, n_features, n_variables)
        y : array_like
            The feature of shape (n_trials,) for estimating task-related O-info
        minsize, maxsize : int | 2, None
            Minimum and maximum size of the multiplets
        method : {'gcmi', 'binning', 'knn'}
            Name of the method to compute entropy. Use either :

                * 'gcmi': gaussian copula entropy [default]
                * 'binning': binning-based estimator of entropy. Note that to
                   use this estimator, the data have be to discretized
                * 'knn': k-nearest neighbor estimator
        kwargs : dict | {}
            Additional arguments are sent to each entropy function
        """
        # ______________________________ INPUTS _______________________________

        # data checking
        data = self._prepare_data(data)

        # check multiplets
        self._prepare_multiplets(minsize, maxsize, y=y)
        self.minsize = max(2, self.minsize)

        logger.info(
            f"Compute the {'task-related ' * self.task_related} HOI "
            f"(min={self.minsize}; max={self.maxsize})"
        )

        # check entropy function
        data, entropy = self._prepare_for_entropy(data, method, y=y, **kwargs)

        # ______________________________ O-INFO _______________________________

        # wrap the entropy function twice to support 4D inputs
        entropy = jax.jit(jax.vmap(jax.vmap(entropy)))

        # use it to compute oinfo
        oinfo_mmult = jax.jit(partial(oinfo_scan, entropy=entropy))

        oinfo = []
        for msize in self:
            combs = self.get_combinations(msize)

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
    ###########################################################################


    x = np.load('/home/etienne/Downloads/data_200_trials', allow_pickle=True)

    logger.setLevel('INFO')

    # x = digitize(x, 8, axis=0)

    model = OinfoZeroLag()
    oinfo = model.fit(
        x[..., 100], method=method, minsize=1, maxsize=None
    )
    # print(model.multiplets)
    # print(model.order)

    lscp = landscape(oinfo.squeeze(), model.order, output='xarray')
    lscp.plot(x='order', y='bins', cmap='turbo', norm=LogNorm())
    plt.axvline(model.undersampling, linestyle='--', color='k')
    plt.title(method, fontsize=24, fontweight='bold')
    plt.show()
