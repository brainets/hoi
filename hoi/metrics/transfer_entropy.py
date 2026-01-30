from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from hoi.metrics.base_hoi import HOIEstimator
from hoi.core.entropies import prepare_for_it
from hoi.core.mi import get_cond_mi
from hoi.utils.progressbar import get_pbar


@partial(jax.jit, static_argnums=(2,))
def _compute_pairwise_te(variables, comb, cmi_fcn=None):
    x_past, y_future, y_past = variables

    x_c = x_past[:, comb[0], jnp.newaxis, :]
    y_c = y_future[:, comb[1], jnp.newaxis, :]
    t_c = y_past[:, comb[1], jnp.newaxis, :]

    return variables, cmi_fcn(x_c, y_c, t_c)


class TransferEntropy(HOIEstimator):
    r"""Pairwise transfer entropy in a dynamical system.

    For each pair of variables, compute the transfer entropy:

    .. math::

        T_{X_i \rightarrow X_j} = I(X_i(t-\tau); X_j(t) | X_j(t-\tau))

    Parameters
    ----------
    x : array_like
        Standard NumPy arrays of shape (n_samples, n_features) or
        (n_samples, n_features, n_variables)
    multiplets : list | None
        List of multiplets to compute. Should be a list of pairs, for
        example [(0, 1), (2, 7)]. By default, all pairs are computed.
    """

    __name__ = "Transfer Entropy"
    _encoding = False
    _positive = "info"
    _negative = "null"
    _symmetric = False
    _directed = True

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
        r"""Compute pairwise transfer entropy.

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
                  callable function written with Jax taking three 2D inputs
                  of shape (n_features, n_samples) and returning a float.

        samples : np.ndarray
            List of samples to use to compute HOI. If None, all samples are
            going to be used.
        tau : int | 1
            The length of the delay to use to compute the transfer entropy.
            Default 1
        direction_axis : {0,2}
            The axis on which to consider the evolution,
            0 for the samples axis, 2 for the variables axis.
            Default 0
        matrix : bool | False
            If True, return a (n_features, n_features, n_variables) matrix.
        kwargs : dict | {}
            Additional arguments are sent to each CMI function

        Returns
        -------
        hoi : array_like
            The NumPy array containing pairwise transfer entropy of
            shape (n_pairs, n_variables), or the full matrix if matrix=True.
        """
        if maxsize is None:
            maxsize = 2
        if maxsize != 2:
            raise ValueError("Transfer entropy is defined for order=2 only.")

        # ________________________________ I/O ________________________________
        # check minsize and maxsize
        minsize, maxsize = self._check_minmax(max(minsize, 2), maxsize)

        # prepare the x for computing cmi
        x, kwargs = prepare_for_it(self._x, method, samples=samples, **kwargs)

        # prepare cmi functions
        cmi_fcn = jax.vmap(get_cond_mi(method=method, **kwargs))
        compute_te = partial(_compute_pairwise_te, cmi_fcn=cmi_fcn)

        # get multiplet indices and order - let to a general format, but max
        # possible order = 2
        h_idx, order = self.get_combinations(minsize, maxsize=maxsize)

        # get progress bar
        pbar = get_pbar(
            iterable=range(order.min(), order.max() + 1), leave=False
        )

        # _______________________________ HOI _________________________________
        offset = 0
        if direction_axis == 2:
            hoi = jnp.zeros(
                (len(order), self.n_variables - tau), dtype=jnp.float32
            )
        else:
            hoi = jnp.zeros((len(order), self.n_variables), dtype=jnp.float32)

        for msize in pbar:
            pbar.set_description(
                desc="Transfer Entropy order %s" % msize, refresh=False
            )

            # combinations of features
            _h_idx = h_idx[order == msize, 0:msize]

            if direction_axis == 0:
                x_past = x[:, :, :-tau]
                y_future = x[:, :, tau:]
                y_past = x[:, :, :-tau]
            elif direction_axis == 2:
                x_past = x[:-tau, :, :]
                y_future = x[tau:, :, :]
                y_past = x[:-tau, :, :]
            else:
                raise ValueError("axis can be eaither equal 0 or 2.")

            # compute pairwise te
            _, _hoi = jax.lax.scan(
                compute_te, (x_past, y_future, y_past), _h_idx
            )

            # fill variables
            n_combs = _h_idx.shape[0]
            hoi = hoi.at[offset : offset + n_combs, :].set(_hoi)

            # updates
            offset += n_combs

        if matrix:
            n_feat = self._n_features_x
            hoi_np = np.asarray(hoi)
            mat = np.zeros((n_feat, n_feat, hoi_np.shape[1]))

            for n_m, mult in enumerate(_h_idx):
                i, j = mult[0], mult[1]
                mat[i, j, :] = hoi_np[n_m, :]

            return mat

        return np.asarray(hoi)
