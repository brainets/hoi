from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from hoi.metrics.base_hoi import HOIEstimator
from hoi.core.entropies import prepare_for_entropy
from hoi.core.mi import get_mi, compute_mi_comb
from hoi.utils.progressbar import get_pbar


class RedundancyMMI(HOIEstimator):
    """Redundancy estimated using the Minimum Mutual Information.

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

    __name__ = "Redundancy MMI"
    _encoding = True
    _positive = "redundancy"
    _negative = "null"
    _symmetric = False

    def __init__(self, x, y, multiplets=None, verbose=None):
        HOIEstimator.__init__(
            self,
            x=x,
            y=y,
            multiplets=multiplets,
            verbose=verbose,
        )

    def fit(self, minsize=2, maxsize=None, method="gcmi", **kwargs):
        """Redundancy Index.

        Parameters
        ----------
        minsize, maxsize : int | 2, None
            Minimum and maximum size of the multiplets
        method : {'gcmi'}
            Name of the method to compute mutual-information. Use either :

                * 'gcmi': gaussian copula MI [default].

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

        # prepare the data for computing mi
        x, kwargs = prepare_for_entropy(self._x, method, **kwargs)
        x, y = self._split_xy(x)

        # prepare mi functions
        mi_fcn = jax.vmap(get_mi(method=method, **kwargs))
        compute_mi = partial(compute_mi_comb, mi=mi_fcn)

        # get multiplet indices and order
        h_idx, order = self.get_combinations(minsize, maxsize=maxsize)

        # get progress bar
        pbar = get_pbar(
            iterable=range(order.min(), order.max() + 1), leave=False
        )

        # _______________________________ HOI _________________________________

        # compute mi I(x_{1}; y), ..., I(x_{n}; y)
        _, i_xiy = jax.lax.scan(
            compute_mi, (x, y), jnp.arange(x.shape[1]).reshape(-1, 1)
        )

        # prepare outputs
        offset = 0
        hoi = jnp.zeros((len(order), self.n_variables), dtype=jnp.float32)
        for msize in pbar:
            pbar.set_description(desc="RedMMI order %s" % msize, refresh=False)

            # combinations of features
            _h_idx = h_idx[order == msize, 0:msize]

            # compute hoi
            _hoi = i_xiy[_h_idx, :].min(1)

            # fill variables
            n_combs = _h_idx.shape[0]
            hoi = hoi.at[offset : offset + n_combs, :].set(_hoi)

            # updates
            offset += n_combs

        return np.asarray(hoi)


if __name__ == "__main__":
    from hoi.utils import get_nbest_mult

    np.random.seed(0)

    x = np.random.rand(200, 7)
    y_red = np.random.rand(x.shape[0])

    # redundancy: (1, 2, 6) + (7, 8)
    x[:, 1] += y_red
    x[:, 2] += y_red
    x[:, 6] += y_red

    model = RedundancyMMI(x, y_red)
    hoi = model.fit(minsize=3, maxsize=5, method="gcmi")
    print(hoi)

    print(get_nbest_mult(hoi, model=model, minsize=3, maxsize=5, n_best=3))
