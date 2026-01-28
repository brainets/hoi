from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from hoi.metrics.base_hoi import HOIEstimator
from hoi.core.entropies import prepare_for_it
from hoi.core.mi import get_mi
from hoi.utils.progressbar import get_pbar


@partial(jax.jit, static_argnums=(2,))
def _compute_pairwise_mi(variables, comb, mi_fcn=None):
    x_c = variables[:, comb[0], np.newaxis, :]
    y_c = variables[:, comb[1], np.newaxis, :]

    return variables, mi_fcn(x_c, y_c)


class MI(HOIEstimator):
    r"""Pairwise mutual information in a dynamical system.

    For each pair of variables, compute the mutual information between past
    and future (phiID-style):

    .. math::

        I((X_i, X_j)_{t-\tau}; (X_i, X_j)_t)

    Parameters
    ----------
    x : array_like
        Standard NumPy arrays of shape (n_samples, n_features) or
        (n_samples, n_features, n_variables)
    multiplets : list | None
        List of multiplets to compute. Should be a list of pairs, for
        example [(0, 1), (2, 7)]. By default, all pairs are computed.
    """

    __name__ = "MI (phiID-style)"
    _encoding = False
    _positive = "info"
    _negative = "null"
    _symmetric = True

    def __init__(self, x, multiplets=None, verbose=None):
        HOIEstimator.__init__(
            self, x=x, multiplets=multiplets, verbose=verbose
        )

    def fit(
        self,
        minsize=2,
        tau=1,
        direction_axis=0,
        maxsize=2,
        method="gc",
        samples=None,
        matrix=False,
        **kwargs,
    ):
        r"""Compute pairwise mutual information.

        Parameters
        ----------
        minsize, maxsize : int | 2, 2
            Minimum and maximum size of the multiplets. Must be 2.
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
                  callable function written with Jax taking two 2D inputs
                  of shape (n_features, n_samples) and returning a float.

        samples : np.ndarray
            List of samples to use to compute HOI. If None, all samples are
            going to be used.
        tau : int | 1
            The length of the delay to use to compute the pairwise MI.
            Default 1
        direction_axis : {0,2}
            The axis on which to consider the evolution,
            0 for the samples axis, 2 for the variables axis.
            Default 0
        matrix : bool | False
            If True, return a (n_features, n_features, n_variables) matrix.
        kwargs : dict | {}
            Additional arguments are sent to each MI function

        Returns
        -------
        hoi : array_like
            The NumPy array containing pairwise mutual information of
            shape (n_pairs, n_variables), or the full matrix if matrix=True.
        """
        if maxsize is None:
            maxsize = 2
        if maxsize != 2:
            raise ValueError("Pairwise MI is defined for order=2 only.")

        # ________________________________ I/O ________________________________
        # check minsize and maxsize
        minsize, maxsize = self._check_minmax(max(minsize, 2), maxsize)

        # prepare the x for computing mi
        x, kwargs = prepare_for_it(self._x, method, samples=samples, **kwargs)

        # prepare mi functions
        mi_fcn = jax.vmap(get_mi(method=method, **kwargs))
        compute_mi = partial(_compute_pairwise_mi, mi_fcn=mi_fcn)

        # get multiplet indices and order - let to a general format, but max
        # possible order = 2

        h_idx, order = self.get_combinations(minsize, maxsize=maxsize)

        # get progress bar
        pbar = get_pbar(
            iterable=range(order.min(), order.max() + 1), leave=False
        )

        # _______________________________ HOI _________________________________
        offset = 0
        hoi = jnp.zeros((len(order), self.n_variables), dtype=jnp.float32)

        for msize in pbar:
            pbar.set_description(
                desc="Pairwise MI order %s" % msize, refresh=False
            )

            # combinations of features
            _h_idx = h_idx[order == msize, 0:msize]

            # compute pairwise mi
            _, _hoi = jax.lax.scan(compute_mi, x, _h_idx)

            # fill variables
            n_combs = _h_idx.shape[0]
            hoi = hoi.at[offset : offset + n_combs, :].set(_hoi)

            # updates
            offset += n_combs

        if matrix:
            n_feat = self._n_features_x
            hoi_np = np.asarray(hoi)
            mat = np.zeros((n_feat, n_feat, hoi_np.shape[1]))

            for n_m, mult in enumerate(self.multiplets):
                i, j = mult[0], mult[1]
                mat[i, j, :] = hoi_np[n_m, :]
                mat[j, i, :] = hoi_np[n_m, :]

            return mat

        return np.asarray(hoi)
