from math import comb as ccomb
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from hoi.metrics.base_hoi import HOIEstimator
from hoi.core.combinatory import combinations
from hoi.core.entropies import prepare_for_entropy
from hoi.core.mi import get_mi, compute_mi_comb
from hoi.utils.progressbar import get_pbar


class InfoTot(HOIEstimator):

    """Total information.

    The total information is the mutual information between set `S` and a
    variable `y`:

    .. math::

        InfoTot(S; Y) = I(x_{1}, ..., x_{n}; y)

    Parameters
    ----------
    x : array_like
        Standard NumPy arrays of shape (n_samples, n_features) or
        (n_samples, n_features, n_variables)
    y : array_like
        The feature of shape (n_samples,)
    multiplets : list | None
        List of multiplets to compute. Should be a list of multiplets, for
        example [(0, 1, 2), (2, 7, 8, 9)]. By default, all multiplets are
        going to be computed.
    """

    __name__ = "Total information"

    def __init__(self, x, y, multiplets=None, verbose=None):
        HOIEstimator.__init__(
            self, x=x, y=y, multiplets=multiplets, verbose=verbose
        )

    def fit(self, minsize=2, maxsize=None, method="gcmi", **kwargs):
        """Compute RSI.

        Parameters
        ----------
        minsize, maxsize : int | 2, None
            Minimum and maximum size of the multiplets
        method : {'gcmi'}
            Name of the method to compute mutual-information. Use either :

                * 'gcmi': gaussian copula MI [default]. See
                  :func:`hoi.core.mi_gcmi_gg`

        kwargs : dict | {}
            Additional arguments are sent to each MI function

        Returns
        -------
        hoi : array_like
            The NumPy array containing values of higher-rder interactions of
            shape (n_multiplets, n_variables)
        """
        # ________________________________ I/O ________________________________
        # check minsize and maxsize
        minsize, maxsize = self._check_minmax(max(minsize, 2), maxsize)

        # prepare the x for computing mi
        x, kwargs = prepare_for_entropy(self._x, method, **kwargs)
        x, y = self._split_xy(x)

        # prepare mi functions
        mi_fcn = jax.vmap(get_mi(method=method, **kwargs))
        compute_mi = partial(compute_mi_comb, mi=mi_fcn)

        # _______________________________ HOI _________________________________

        # get progress bar
        pbar = get_pbar(iterable=range(minsize, maxsize + 1), leave=False)

        # prepare the shapes of outputs
        n_mults = sum(
            [ccomb(self._n_features_x, c) for c in range(minsize, maxsize + 1)]
        )
        hoi = jnp.zeros((n_mults, self.n_variables), dtype=jnp.float32)
        h_idx = jnp.full((n_mults, maxsize), -1, dtype=int)
        order = jnp.zeros((n_mults,), dtype=int)

        offset = 0
        for msize in pbar:
            pbar.set_description(
                desc="Infotot order %s" % msize, refresh=False
            )

            # get combinations
            _h_idx = combinations(self._n_features_x, msize, astype="jax")
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

    model = InfoTot(x, y)
    # hoi = model.fit(minsize=2, maxsize=6, method='kernel')
    hoi = model.fit(minsize=2, maxsize=6, method="gcmi")

    print(get_nbest_mult(hoi, model, minsize=3, maxsize=3))

    plot_landscape(
        hoi,
        model,
        kind="scatter",
        undersampling=False,
        plt_kwargs=dict(cmap="turbo"),
    )
    plt.show()
