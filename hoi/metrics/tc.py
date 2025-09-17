from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from hoi.core.entropies import get_entropy, prepare_for_it
from hoi.metrics.base_hoi import HOIEstimator
from hoi.utils.progressbar import get_pbar


@partial(jax.jit, static_argnums=(2, 3))
def _tc_no_ent(inputs, index, entropy_3d=None, entropy_4d=None):
    data, acc = inputs

    # tuple selection
    x_c = data[:, index, :]

    # compute h(x^{n})
    h_xn = entropy_3d(x_c)

    # compute \sum_{j=1}^{n} h(x_{j})
    h_xj_sum = entropy_4d(x_c[:, :, jnp.newaxis, :]).sum(0)

    # compute tc
    tc = h_xj_sum - h_xn

    return inputs, tc


class TC(HOIEstimator):
    r"""Total correlation.

    Total correlation is the oldest extension of mutual information to
    an arbitrary number of variables. It is defined as:

    .. math::

        TC(X^{n})  &=  \sum_{j=1}^{n} H(X_{j}) - H(X^{n}) \\

    The total correlation is equivalent to the Kullback-Liebler divergence
    between the joint distribution :math: `P(X)` and the product of the
    marginals. The total correlation is largely a measure of redundancy,
    sensitive to information shared between single elements.

    Parameters
    ----------
    x : array_like
        Standard NumPy arrays of shape (n_samples, n_features) or
        (n_samples, n_features, n_variables)
    y : array_like
        The feature of shape (n_samples,) for estimating task-related TC
    multiplets : list | None
        List of multiplets to compute. Should be a list of multiplets, for
        example [(0, 1, 2), (2, 7, 8, 9)]. By default, all multiplets are
        going to be computed.

    References
    ----------
    Watabe, 1960 :cite:`watanabe1960information`
    """

    __name__ = "Total correlation"
    _encoding = False
    _negative = "null"
    _positive = "null"
    _symmetric = True

    def __init__(self, x, y=None, multiplets=None, verbose=None):
        HOIEstimator.__init__(
            self, x=x, y=y, multiplets=multiplets, verbose=verbose
        )

    def fit(
        self, minsize=2, maxsize=None, method="gc", samples=None, **kwargs
    ):
        """Compute the Total correlation.

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
        # ________________________________ I/O ________________________________
        # check min and max sizes
        minsize, maxsize = self._check_minmax(minsize, maxsize)

        # prepare the x for computing entropy
        x, kwargs = prepare_for_it(self._x, method, samples=samples, **kwargs)

        # get entropy function
        entropy = jax.vmap(get_entropy(method=method, **kwargs))
        tc_no_ent = partial(
            _tc_no_ent,
            entropy_3d=entropy,
            entropy_4d=jax.vmap(entropy, in_axes=1),
        )

        # get multiplet indices and order
        h_idx, order = self.get_combinations(minsize, maxsize=maxsize)

        # get progress bar
        pbar = get_pbar(
            iterable=range(order.min(), order.max() + 1), leave=False
        )

        # ______________________________ ENTROPY ____________________________
        offset = 0
        hoi = jnp.zeros((len(order), self.n_variables), dtype=jnp.float32)
        for msize in pbar:
            pbar.set_description(desc="TC (%i)" % msize, refresh=False)

            # get the number of features when considering y
            n_feat_xy = msize + self._n_features_y

            # combinations of features
            _h_idx = h_idx[order == msize, 0:n_feat_xy]

            # indices for X_{-j} and skip first column
            acc = jnp.mgrid[0:n_feat_xy, 0:n_feat_xy].sum(0) % n_feat_xy
            acc = acc[:, 1:]

            # compute tc
            _, _hoi = jax.lax.scan(tc_no_ent, (x, acc), _h_idx)

            # fill variables
            n_combs = _h_idx.shape[0]
            hoi = hoi.at[offset : offset + n_combs, :].set(_hoi)

            # updates
            offset += n_combs

        return np.asarray(hoi)
