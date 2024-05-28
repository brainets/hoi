from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from hoi.metrics.base_hoi import HOIEstimator
from hoi.core.entropies import prepare_for_entropy
from hoi.core.mi import get_mi, compute_mi_comb
from hoi.utils.progressbar import get_pbar


@partial(jax.jit, static_argnums=(2,))
def _compute_phi_syn(inputs, comb, mi_fcn=None):
    x, y, ind = inputs

    # select combination
    x_c = x[:, :, :]
    y_c = y[:, comb, :]

    # compute info tot I({x_{1}, ..., x_{n}}; S)
    _, i_tot = mi_fcn((x_c, y_c), comb)

    # compute max(I(x_{-j}; S))
    _, i_maxj = jax.lax.scan(mi_fcn, (x_c[:, comb, :], y_c), ind)

    return inputs, i_tot - i_maxj.max(0)


class SynergyphiID(HOIEstimator):
    r"""Synergy (phiID).

    For each couple of variable the synergy about their future as in
    Luppi et al (2022), using the Minimum Mutual Information (MMI) approach:

    .. math::

        Syn(X,Y) =  I(X_{t-\tau},Y_{t-\tau};X_{t},Y_t) -
                            max \{ I(X_{t-\tau};X_t,Y_t),
                            I(Y_{t-\tau};X_t,Y_t) \}

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
    Luppi et al, 2022 :cite:`luppi2022synergistic`
    """

    __name__ = "Synergy phiID MMI"
    _encoding = False
    _positive = "synergy"
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
        r"""Synergy (phiID).

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
            The axis on which to consider the evolution,
            0 for the samples axis, 2 for the variables axis.
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

        # prepare the x for computing mi
        x, kwargs = prepare_for_entropy(self._x, method, **kwargs)

        # prepare mi functions
        mi_fcn = jax.vmap(get_mi(method=method, **kwargs))
        compute_mi = partial(compute_mi_comb, mi=mi_fcn)
        compute_syn = partial(_compute_phi_syn, mi_fcn=compute_mi)

        # get multiplet indices and order
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
            pbar.set_description(desc="SynMMI order %s" % msize, refresh=False)

            # combinations of features
            _h_idx = h_idx[order == msize, 0:msize]

            # define indices for I(x_{-j}; S)
            ind = (jnp.mgrid[0:msize, 0:msize].sum(0) % msize)[:, 1:]

            if direction_axis == 0:
                x_c = x[:, :, :-tau]
                y = x[:, :, tau:]

            elif direction_axis == 2:
                x_c = x[:-tau, :, :]
                y = x[tau:, :, :]

            else:
                raise ValueError("axis can be eaither equal 0 or 2.")

            # compute hoi
            _, _hoi = jax.lax.scan(compute_syn, (x_c, y, ind), _h_idx)

            # fill variables
            n_combs = _h_idx.shape[0]
            hoi = hoi.at[offset : offset + n_combs, :].set(_hoi)

            # updates
            offset += n_combs

        return np.asarray(hoi)
