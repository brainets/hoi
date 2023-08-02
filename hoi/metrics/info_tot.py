from math import comb as ccomb

from functools import partial

import logging

import numpy as np

import jax
import jax.numpy as jnp

from hoi.metrics.base_hoi import HOIEstimator
from hoi.core.combinatory import combinations
from hoi.core.entropies import get_entropy, prepare_for_entropy
from hoi.core.mi import mi_entr_comb
from hoi.utils.progressbar import get_pbar

logger = logging.getLogger("hoi")


class InfoTot(HOIEstimator):

    """Total information.

    The total information is the mutual information between set `S` and a
    variable `y`:

    .. math::

        InfoTot(S; Y) = I(x_{1}, ..., x_{n}; y)

    Parameters
    ----------
    data : array_like
        Standard NumPy arrays of shape (n_samples, n_features) or
        (n_samples, n_features, n_variables)
    y : array_like
        The feature of shape (n_trials,) for estimating task-related O-info
    """

    __name__ = "Total information"

    def __init__(self, data, y, verbose=None):
        HOIEstimator.__init__(self, data=data, y=y, verbose=verbose)

    def fit(self, minsize=2, maxsize=None, method="gcmi", **kwargs):
        """Compute RSI.

        Parameters
        ----------
        minsize, maxsize : int | 2, None
            Minimum and maximum size of the multiplets
        method : {'gcmi', 'binning', 'knn', 'kernel}
            Name of the method to compute entropy. Use either :

                * 'gcmi': gaussian copula entropy [default]. See
                  :func:`hoi.core.entropy_gcmi`
                * 'binning': binning-based estimator of entropy. Note that to
                  use this estimator, the data have be to discretized. See
                  :func:`hoi.core.entropy_bin`
                * 'knn': k-nearest neighbor estimator. See
                  :func:`hoi.core.entropy_knn`
                * 'kernel': kernel-based estimator of entropy
                  see :func:`hoi.core.entropy_kernel`

        kwargs : dict | {}
            Additional arguments are sent to each entropy function
        """
        # ________________________________ I/O ________________________________
        # check minsize and maxsize
        minsize, maxsize = self._check_minmax(max(minsize, 2), maxsize)

        # prepare the data for computing entropy
        data, kwargs = prepare_for_entropy(self._data, method, **kwargs)
        x, y = data[:, 0:-1, :], data[:, [-1], :]

        # prepare entropy functions
        entropy = jax.vmap(get_entropy(method=method, **kwargs))
        compute_mi = partial(mi_entr_comb, entropy=entropy)

        # _______________________________ HOI _________________________________

        # get progress bar
        pbar = get_pbar(iterable=range(minsize, maxsize + 1), leave=False)

        # prepare the shapes of outputs
        # E501
        x = self.n_features
        t = [ccomb(x - 1, c) for c in range(minsize, maxsize + 1)]
        n_mults = sum(t)
        hoi = jnp.zeros((n_mults, self.n_variables), dtype=jnp.float32)
        h_idx = jnp.full((n_mults, maxsize), -1, dtype=int)
        order = jnp.zeros((n_mults,), dtype=int)

        offset = 0
        for msize in pbar:
            x = msize
            pbar.set_description(desc="Infotot order %s" % x, refresh=False)

            # get combinations
            _h_idx = combinations(
                self.n_features - 1, msize, as_iterator=False, as_jax=True
            )
            n_combs, n_feat = _h_idx.shape
            sl = slice(offset, offset + n_combs)

            # fill indices and order
            h_idx = h_idx.at[sl, 0:n_feat].set(_h_idx)
            order = order.at[sl].set(msize)

            # compute I({x_{1}, ..., x_{n}}; S)
            _, _hoi = jax.lax.scan(compute_mi, (x, y), _h_idx)
            hoi = hoi.at[sl, :].set(_hoi)

            # updates
            offset += n_combs

        self._order = order
        self._multiplets = h_idx
        self._keep = np.ones_like(self._order, dtype=bool)

        return np.asarray(hoi)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from hoi.utils import get_nbest_mult
    from hoi.plot import plot_landscape

    plt.style.use("ggplot")

    x = np.random.rand(200, 7)
    # y = x[:, 0]
    # y[100::] = x[100::, 1]
    y = x[:, 0] + x[:, 3] + x[:, 5]

    # y = x[:, 4]
    # x[:, 5] += y

    from sklearn.preprocessing import KBinsDiscretizer

    x = (
        KBinsDiscretizer(3, "ordinal", strategy="uniform", subsample=None)
        .fit_transform(x)
        .astype(int)
    )
    y = (
        KBinsDiscretizer(3, "ordinal", strategy="uniform", subsample=None)
        .fit_transform(y.reshape(-1, 1))
        .astype(int)
        .squeeze()
    )

    model = InfoTot(x, y)
    # hoi = model.fit(minsize=2, maxsize=6, method='kernel')
    hoi = model.fit(minsize=2, maxsize=6, method="binning")

    print(get_nbest_mult(hoi, model, minsize=3, maxsize=3))

    t = dict(cmap="turbo")
    s = "scatter"
    plot_landscape(hoi, model, kind=s, undersampling=False, plt_kwargs=t)
    plt.show()
