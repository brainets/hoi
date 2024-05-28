from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from hoi.metrics.base_hoi import HOIEstimator
from hoi.core.entropies import prepare_for_entropy
from hoi.core.mi import get_mi, compute_mi_comb_phi
from hoi.utils.progressbar import get_pbar


@partial(jax.jit, static_argnums=(2,))
def _compute_phiid_red(inputs, comb, mi_fcn=None):
    x, y, ind = inputs

    # select combination
    x_c = x[:, comb, :]
    y_c = y[:, comb, :]

    # compute max(I(x_{-j}; S))
    _, i_minj = jax.lax.scan(mi_fcn, (x_c, y_c), ind)

    return inputs, i_minj.min(0)


class RedundancyphiID(HOIEstimator):
    r"""Redundancy (phiID).

    Estimated using the Minimum Mutual Information (MMI) as follow:

    .. math::

        Red(X,Y) =   min \{ I(X_{t- \tau};X_t), I(X_{t-\tau};Y_t),
                            I(Y_{t-\tau}; X_t), I(Y_{t-\tau};Y_t) \}

    Parameters
    ----------
    x : array_like
        Standard NumPy arrays of shape (n_samples, n_features) or
        (n_samples, n_features, n_variables)
    multiplets : list | None
        List of multiplets to compute. Should be a list of multiplets, for
        example [(0, 1), (2, 7)]. By default, all multiplets are
        going to be computed.

    References
    ----------
    Pedro AM Mediano et al, 2021 :cite:`mediano2021towards`
    """

    __name__ = "Redundancy phiID MMI"
    _encoding = False
    _positive = "redundancy"
    _negative = "null"
    _symmetric = True

    def __init__(self, x, multiplets=None, verbose=None):
        HOIEstimator.__init__(
            self,
            x=x,
            multiplets=multiplets,
            verbose=verbose,
        )

    def fit(
        self,
        minsize=2,
        tau=1,
        direction_axis=0,
        maxsize=None,
        method="gcmi",
        **kwargs
    ):
        """Redundancy (phiID).

        Parameters
        ----------
        minsize, maxsize : int | 2, None
            Minimum and maximum size of the multiplets
        method : {'gcmi'}
            Name of the method to compute mutual-information. Use either :

                * 'gcmi': gaussian copula entropy [default]. See
                  :func:`hoi.core.entropy_gcmi`
                * 'binning': binning-based estimator of entropy. Note that to
                  use this estimator, the data have be to discretized. See
                  :func:`hoi.core.entropy_bin`
                * 'knn': k-nearest neighbor estimator. See
                  :func:`hoi.core.entropy_knn`
                * 'kernel': kernel-based estimator of entropy
                  see :func:`hoi.core.entropy_kernel`

        tau : int | 1
            The length of the delay to use to compute the redundancy as
            defined in the phiID.
            Default 1
        direction_axis : {0,2}
            the axis on which to consider the evolution.
            0 for the samples axis, 2 for the variables axis
            Default 0
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

        # prepare mi functions
        mi_fcn = jax.vmap(get_mi(method=method, **kwargs))
        compute_mi = partial(compute_mi_comb_phi, mi=mi_fcn)
        compute_phiid_red = partial(_compute_phiid_red, mi_fcn=compute_mi)

        # get multiplet indices and order
        h_idx, order = self.get_combinations(minsize, maxsize=maxsize)

        # get progress bar
        pbar = get_pbar(
            iterable=range(order.min(), order.max() + 1), leave=False
        )

        # _______________________________ HOI _________________________________

        if direction_axis == 0:
            x_c = x[:, :, :-tau]
            y = x[:, :, tau:]

        elif direction_axis == 2:
            x_c = x[:-tau, :, :]
            y = x[tau:, :, :]

        # prepare outputs
        offset = 0
        if direction_axis == 2:
            hoi = jnp.zeros(
                (len(order), self.n_variables - tau), dtype=jnp.float32
            )
        else:
            hoi = jnp.zeros((len(order), self.n_variables), dtype=jnp.float32)

        for msize in pbar:
            pbar.set_description(desc="RedMMI order %s" % msize, refresh=False)

            # combinations of features
            _h_idx = h_idx[order == msize, 0:msize]

            dd = jnp.array(np.meshgrid(jnp.arange(msize), jnp.arange(msize))).T
            ind = dd.reshape(-1, 2, 1)

            # compute hoi
            _, _hoi = jax.lax.scan(compute_phiid_red, (x_c, y, ind), _h_idx)

            # fill variables
            n_combs = _h_idx.shape[0]
            hoi = hoi.at[offset : offset + n_combs, :].set(_hoi)

            # updates
            offset += n_combs

        return np.asarray(hoi)
