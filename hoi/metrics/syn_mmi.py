from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from hoi.core.entropies import prepare_for_it
from hoi.core.mi import compute_mi_comb, get_mi
from hoi.metrics.base_hoi import HOIEstimator
from hoi.utils.progressbar import get_pbar


@partial(jax.jit, static_argnums=(2,))
def _compute_syn(inputs, comb, mi_fcn=None):
    x, y, ind = inputs

    # compute info tot I({x_{1}, ..., x_{n}}; S)
    _, i_tot = mi_fcn((x, y), comb)

    # select combination
    x_c = x[:, comb, :]

    # compute max(I(x_{-j}; S))
    _, i_maxj = jax.lax.scan(mi_fcn, (x_c, y), ind)

    return inputs, i_tot - i_maxj.max(0)


class SynergyMMI(HOIEstimator):
    """Synergy estimated using the Minimum Mutual Information.

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

    __name__ = "Synergy MMI"
    _encoding = True
    _positive = "synergy"
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

    def fit(
        self, minsize=2, maxsize=None, method="gc", samples=None, **kwargs
    ):
        """Synergy Index.

        Parameters
        ----------
        minsize, maxsize : int | 2, None
            Minimum and maximum size of the multiplets
        method : {'gc', 'binning', 'knn', 'kernel', callable}
            Name of the method to compute entropy. Use either :

                * 'gc': gaussian copula entropy [default]. See
                  :func:`hoi.core.entropy_gc`
                * 'gauss': gaussian entropy. See :func:`hoi.core.entropy_gauss`
                * 'binning': binning-based estimator of entropy. Note that to
                  use this estimator, the data have be to discretized. See
                  :func:`hoi.core.entropy_bin`
                * 'knn': k-nearest neighbor estimator. See
                  :func:`hoi.core.entropy_knn`
                * 'kernel': kernel-based estimator of entropy
                  see :func:`hoi.core.entropy_kernel`
                * A custom entropy estimator can be provided. It should be a
                  callable function written with Jax taking a single 2D input
                  of shape (n_features, n_samples) and returning a float.

        samples : np.ndarray
            List of samples to use to compute HOI. If None, all samples are
            going to be used.
        kwargs : dict | {}
            Additional arguments are sent to each MI function

        Returns
        -------
        hoi : array_like
            The NumPy array containing values of higher-order interactions of
            shape (n_multiplets, n_variables)
        """
        # ________________________________ I/O ________________________________
        # check minsize and maxsize
        minsize, maxsize = self._check_minmax(max(minsize, 2), maxsize)

        # prepare the x for computing mi
        x, kwargs = prepare_for_it(self._x, method, samples=samples, **kwargs)
        x, y = self._split_xy(x)

        # prepare mi functions
        mi_fcn = jax.vmap(get_mi(method=method, **kwargs))
        compute_mi = partial(compute_mi_comb, mi=mi_fcn)
        compute_syn = partial(_compute_syn, mi_fcn=compute_mi)

        # get multiplet indices and order
        h_idx, order = self.get_combinations(minsize, maxsize=maxsize)

        # get progress bar
        pbar = get_pbar(
            iterable=range(order.min(), order.max() + 1), leave=False
        )

        # _______________________________ HOI _________________________________

        offset = 0
        hoi = jnp.zeros((len(order), self.n_variables), dtype=jnp.float32)
        for msize in pbar:
            pbar.set_description(desc="SynMMI order %s" % msize, refresh=False)

            # combinations of features
            _h_idx = h_idx[order == msize, 0:msize]

            # define indices for I(x_{-j}; S)
            ind = (jnp.mgrid[0:msize, 0:msize].sum(0) % msize)[:, 1:]

            # compute hoi
            _, _hoi = jax.lax.scan(compute_syn, (x, y, ind), _h_idx)

            # fill variables
            n_combs = _h_idx.shape[0]
            hoi = hoi.at[offset : offset + n_combs, :].set(_hoi)

            # updates
            offset += n_combs

        return np.asarray(hoi)
