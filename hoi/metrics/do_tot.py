from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from hoi.core.entropies import prepare_for_it
from hoi.core.mi import (
    compute_cmi_comb,
    compute_mi_doinfo_sub,
    compute_mi_doinfo_tot,
    get_cond_mi,
)
from hoi.metrics.base_hoi import HOIEstimator
from hoi.utils.progressbar import get_pbar


@partial(jax.jit, static_argnums=(2, 3))
def compute_dyn_oinfo(inputs, comb, cmi_fcn_tot=None, cmi_fcn_sub=None):
    x, y, ind, ind_sub = inputs

    n = len(ind)

    # select combination
    x_c = x[:, comb, :]
    y_c = y[:, comb, :]

    # compute info tot I({x_{1}, ..., x_{n}}; S)
    _, i_tot = jax.lax.scan(cmi_fcn_tot, (x_c, y_c, ind), jnp.arange(n))

    # compute max(I(x_{-j}; S))
    _, i_subj = jax.lax.scan(
        cmi_fcn_sub, (x_c, y_c, ind, ind_sub), jnp.arange(n)
    )

    return inputs, (2 - n) * i_tot.sum(0) + i_subj.sum(0)


class DOtot(HOIEstimator):
    r"""The total dynamic O-information (dOtot).

    For each multiplet of size n, the dO is defined as follows:

    .. math::

        dO_{tot}(X^n) =  \sum_{j=1}^{n} dO_j(X^n)

    where
    .. math::

        dO_j(X^n) = & (1-n)I(X_{-j}^n(t-\tau); X_{j}(t) | X_{-j}^n(t-\tau))-\\
        &- \sum_{i \in X_{-j}^n} I(X_{-ij}^n(t-\tau); X_{j}(t) | X_{j}(t-\tau))

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
    Stramaglia et al, 2022 :cite:`stramaglia2021quantifying` and Robiglio
    et al, 2025 :cite:`robiglio2025synergistic`
    """

    __name__ = "Dynamic O-information total"
    _encoding = False
    _positive = "redundancy"
    _negative = "synergy"
    _symmetric = False

    def __init__(self, x, multiplets=None, verbose=None):
        HOIEstimator.__init__(
            self, x=x, multiplets=multiplets, verbose=verbose
        )

    def fit(
        self,
        minsize=3,
        tau=1,
        direction_axis=0,
        maxsize=None,
        method="gc",
        samples=None,
        **kwargs,
    ):
        r"""Synergy (phiID).

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
            The NumPy array containing values of higher-order interactions of
            shape (n_multiplets, n_variables)
        """
        # ________________________________ I/O ________________________________
        # check minsize and maxsize
        minsize, maxsize = self._check_minmax(max(minsize, 2), maxsize)

        # prepare the x for computing mi
        x, kwargs = prepare_for_it(self._x, method, samples=samples, **kwargs)

        # prepare mi functions
        cmi_fcn = jax.vmap(get_cond_mi(method=method, **kwargs))
        compute_dyn_otot = partial(compute_mi_doinfo_tot, cmi=cmi_fcn)

        cmi_fcn_s = partial(compute_cmi_comb, cmi=cmi_fcn)
        compute_dyn_osub = partial(compute_mi_doinfo_sub, cmi=cmi_fcn_s)

        compute_do = partial(
            compute_dyn_oinfo,
            cmi_fcn_tot=compute_dyn_otot,
            cmi_fcn_sub=compute_dyn_osub,
        )
        # compute_do = partial(compute_dyn_oinfo, mi_fcn=compute_mi)

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
            pbar.set_description(desc="dO_tot order %s" % msize, refresh=False)

            # combinations of features
            _h_idx = h_idx[order == msize, 0:msize]

            # define indices for I(x_{-j}; S)
            ind = (jnp.mgrid[0:msize, 0:msize].sum(0) % msize)[:, 1:]
            ind_sub = (
                jnp.mgrid[0 : msize - 1, 0 : msize - 1].sum(0) % (msize - 1)
            )[:, 1:]

            if direction_axis == 0:
                x_c = x[:, :, :-tau]
                y = x[:, :, tau:]

            elif direction_axis == 2:
                x_c = x[:-tau, :, :]
                y = x[tau:, :, :]

            else:
                raise ValueError("axis can be eaither equal 0 or 2.")

            # compute hoi
            _, _hoi = jax.lax.scan(compute_do, (x_c, y, ind, ind_sub), _h_idx)

            # fill variables
            n_combs = _h_idx.shape[0]
            hoi = hoi.at[offset : offset + n_combs, :].set(_hoi)

            # updates
            offset += n_combs

        return np.asarray(hoi)
