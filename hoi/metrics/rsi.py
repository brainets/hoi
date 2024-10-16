from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from hoi.metrics.base_hoi import HOIEstimator
from hoi.core.entropies import prepare_for_it
from hoi.core.mi import get_mi, compute_mi_comb
from hoi.utils.progressbar import get_pbar


class RSI(HOIEstimator):
    """Redundancy-Synergy Index (RSI).

    The RSI is designed to be maximal and positive when the variables in S are
    purported to provide synergistic information about Y. It should be negative
    when the variables in S provide redundant information about Y. It is
    defined as the total information carried by a set `S` exceeding the
    information provided by individual elements of S:

    .. math::

        RSI(S; Y) \equiv I(S; Y) - \sum_{x_{i}\in S} I(x_{i}; Y)

    with :

    .. math::

        S = x_{1}, ..., x_{n}

    The RSI has been referred to as the SynSum :cite:`globerson2009minimum`,
    the WholeMinusSum synergy :cite:`griffith2014quantifying`, and
    the negative of the redundancy-synergy index has also been
    referred to as the redundancy :cite:`schneidman2003network`.

    .. warning::

        * :math:`RSI(S; Y) > 0 \Rightarrow Synergy`
        * :math:`RSI(S; Y) < 0 \Rightarrow Redundancy`

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

    References
    ----------
    Chechik et al. 2001 :cite:`chechik2001group`; Timme et al., 2014
    :cite:`timme2014synergy`
    """

    __name__ = "Redundancy-Synergy Index"
    _encoding = True
    _positive = "synergy"
    _negative = "redundancy"
    _symmetric = True

    def __init__(self, x, y, multiplets=None, verbose=None):
        HOIEstimator.__init__(
            self, x=x, y=y, multiplets=multiplets, verbose=verbose
        )

    def fit(
        self, minsize=2, maxsize=None, method="gc", samples=None, **kwargs
    ):
        """Compute RSI.

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

        # get multiplet indices and order
        h_idx, order = self.get_combinations(minsize, maxsize=maxsize)

        # get progress bar
        pbar = get_pbar(
            iterable=range(order.min(), order.max() + 1), leave=False
        )

        # _______________________________ HOI _________________________________

        # compute mi I(x_{1}y y), ..., I(x_{n}; y)
        _, i_xiy = jax.lax.scan(
            compute_mi, (x, y), jnp.arange(x.shape[1]).reshape(-1, 1)
        )

        offset = 0
        hoi = jnp.zeros((len(order), self.n_variables), dtype=jnp.float32)
        for msize in pbar:
            pbar.set_description(desc="RSI order %s" % msize, refresh=False)

            # combinations of features
            _h_idx = h_idx[order == msize, 0:msize]

            # compute I({x_{1}, ..., x_{n}}; S)
            _, _i_xy = jax.lax.scan(compute_mi, (x, y), _h_idx)

            # compute hoi
            _hoi = _i_xy - i_xiy[_h_idx, :].sum(1)

            # fill variables
            n_combs = _h_idx.shape[0]
            hoi = hoi.at[offset : offset + n_combs, :].set(_hoi)

            # updates
            offset += n_combs

        return np.asarray(hoi)
