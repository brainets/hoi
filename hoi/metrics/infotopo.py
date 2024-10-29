import warnings

import numpy as np

import jax
import jax.numpy as jnp

from hoi.metrics.base_hoi import HOIEstimator
from jax_tqdm import scan_tqdm


###############################################################################
###############################################################################
#                            MUTUAL INFORMATION
###############################################################################
###############################################################################


# @jax.jit
# def find_indices(inputs, c):
#     combs, keep = inputs

#     keep = jnp.add(keep, (combs == c).any(1).astype(int))

#     return (combs, keep), None


@jax.jit
def compute_mi(inputs, iterators):
    combs, h, order = inputs
    _, comb = iterators

    # scanning over indices
    # is_inside = jnp.zeros((combs.shape[0],), dtype=int)
    # (_, is_inside), _ = jax.lax.scan(find_indices, (combs, is_inside), comb)

    # tensor implementation
    is_inside = (combs == comb[jnp.newaxis, ...]).sum((1, 2))

    # compute infotopo
    info = jnp.sum(h, where=(is_inside == order).reshape(-1, 1), axis=0)

    return inputs, info


class InfoTopo(HOIEstimator):
    """Topological Information.

    The multivariate mutual information :math:`I_{k}` quantify the
    variability/randomness and the statistical dependences between variables
    is defined by :

    .. math::

        I_{k}(X_{1}; ...; X_{k}) = \sum_{i=1}^{k} (-1)^{i - 1} \sum_{
         I\subset[k];card(I)=i} H_{i}(X_{I})

    .. warning::

        * :math:`I_{k}(X_{1}; ...; X_{k}) > 0 \Rightarrow Redundancy`
        * :math:`I_{k}(X_{1}; ...; X_{k}) < 0 \Rightarrow Synergy`

    Parameters
    ----------
    x : array_like
        Standard NumPy arrays of shape (n_samples, n_features) or
        (n_samples, n_features, n_variables)

    References
    ----------
    Baudot et al., 2019 :cite:`baudot2019infotopo`; Tapia et al., 2018
    :cite:`tapia2018neurotransmitter`
    """

    __name__ = "Topological Information"
    _encoding = False
    _positive = "redundancy"
    _negative = "synergy"
    _symmetric = True

    def __init__(self, x, y=None, verbose=None):
        # for infotopo, the multiplets are set to None because this metric
        # first require to compute entropies and then associate them to form
        # the MI. Same for the target y.
        if y is not None:
            warnings.warn("For InfoTopo, y input is going to be ignored.")
        HOIEstimator.__init__(
            self, x=x, y=None, multiplets=None, verbose=verbose
        )

    def fit(
        self, minsize=1, maxsize=None, method="gc", samples=None, **kwargs
    ):
        """Compute Topological Information.

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
            Additional arguments are sent to each entropy function

        Returns
        -------
        hoi : array_like
            The NumPy array containing values of higher-order interactions of
            shape (n_multiplets, n_variables)
        """
        # ____________________________ ENTROPIES ______________________________

        minsize, maxsize = self._check_minmax(minsize, maxsize)
        h_x, h_idx, order = self.compute_entropies(
            minsize=1,
            maxsize=maxsize,
            method=method,
            samples=samples,
            **kwargs,
        )

        # _______________________________ HOI _________________________________

        # compute order and multiply entropies
        h_x_sgn = jnp.multiply(((-1.0) ** (order.reshape(-1, 1) - 1)), h_x)

        # subselection of multiplets
        mults, _ = self.get_combinations(minsize, maxsize=maxsize)
        h_idx_2 = jnp.where(mults == -1, -2, mults)
        n_mult = mults.shape[0]

        # progress-bar definition
        pbar = scan_tqdm(n_mult, message="Mutual information")

        # compute mi
        _, hoi = jax.lax.scan(
            pbar(compute_mi),
            (h_idx[..., jnp.newaxis], h_x_sgn, order),
            (jnp.arange(n_mult), h_idx_2),
        )

        return np.asarray(hoi)
