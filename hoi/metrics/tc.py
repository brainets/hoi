from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from hoi.metrics.base_hoi import HOIEstimator
from hoi.core.entropies import get_entropy, prepare_for_entropy
from hoi.utils.progressbar import get_pbar


@partial(jax.jit, static_argnums=(2, 3))
def _tc_no_ent(inputs, index, entropy_3d=None, entropy_4d=None):
    data, acc = inputs
    msize = len(index)

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

    Total correlation is the oldest exstension of mutual information to
    an arbitrary number of variables. It is defined as:

    .. math::

        \TC(X^{n})  &=  \sum_{j=1}^{n} H(X_{j}) - H(X^{n}) \\

    The total correlation is equivalent to the Kullback-Liebler
    vergence between the joint distribution :math: `P(X)` and the product
    of the marginals. The total correlation is largely a measure
    of redundancy, sensitive to information shared between single
    elements.

    Parameters
    ----------
    x : array_like
        Standard NumPy arrays of shape (n_samples, n_features) or
        (n_samples, n_features, n_variables)
    y : array_like
        The feature of shape (n_trials,) for estimating task-related TC
    multiplets : list | None
        List of multiplets to compute. Should be a list of multiplets, for
        example [(0, 1, 2), (2, 7, 8, 9)]. By default, all multiplets are
        going to be computed.

    References
    ----------
    Watabe, 1960 :cite:`watabe1960information`
    """

    __name__ = "Total correlation"

    def __init__(self, x, y=None, multiplets=None, verbose=None):
        HOIEstimator.__init__(
            self, x=x, y=y, multiplets=multiplets, verbose=verbose
        )

    def fit(self, minsize=2, maxsize=None, method="gcmi", **kwargs):
        """Compute the Total correlation.

        Parameters
        ----------
        minsize, maxsize : int | 2, None
            Minimum and maximum size of the multiplets
        method : {'gcmi', 'binning', 'knn', 'kernel}
            Name of the method to compute entropy. Use either :

                * 'gcmi': gaussian copula entropy [default]. See
                  :func:`hoi.core.entropy_gcmi`
                * 'binning': binning-based estimator of entropy. Note that to
                  use this estimator, the data have be to discretized. See
                  :func:`hoi.core.entropy_bin`
                * 'knn': k-nearest neighbor estimator. See
                  :func:`hoi.core.entropy_knn`
                * 'kernel': kernel-based estimator of entropy
                  see :func:`hoi.core.entropy_kernel`

        kwargs : dict | {}
            Additional arguments are sent to each entropy function
        """
        # ________________________________ I/O ________________________________
        # check min and max sizes
        minsize, maxsize = self._check_minmax(minsize, maxsize)

        # prepare the x for computing entropy
        x, kwargs = prepare_for_entropy(self._x, method, **kwargs)

        # get entropy function
        entropy = jax.vmap(get_entropy(method=method, **kwargs))
        tc_no_ent = partial(
            _tc_no_ent,
            entropy_3d=entropy,
            entropy_4d=jax.vmap(entropy, in_axes=1),
        )

        # prepare output
        kw_combs = dict(maxsize=maxsize, astype="jax")
        h_idx = self.get_combinations(minsize, **kw_combs)
        order = self.get_combinations(minsize, order=True, **kw_combs)

        # subselection of multiplets
        self._multiplets = self.filter_multiplets(h_idx, order)
        order = (self._multiplets >= 0).sum(1)

        # get progress bar
        pbar = get_pbar(
            iterable=range(order.min(), order.max() + 1), leave=False
        )

        # ______________________________ ENTROPY ____________________________
        offset = 0
        hoi = jnp.zeros((len(order), self.n_variables), dtype=jnp.float32)
        for msize in pbar:
            pbar.set_description(desc="TC (%i)" % msize, refresh=False)

            # combinations of features
            keep = order == msize
            _h_idx = self._multiplets[keep, 0:msize]

            # generate indices for accumulated entropies
            acc = jnp.mgrid[0:msize, 0:msize].sum(0) % msize

            # compute tc
            _, _hoi = jax.lax.scan(tc_no_ent, (x, acc[:, 1:]), _h_idx)

            # fill variables
            n_combs, n_feat = _h_idx.shape
            hoi = hoi.at[offset : offset + n_combs, :].set(_hoi)

            # updates
            offset += n_combs

        return np.asarray(hoi)


if __name__ == "__main__":
    import numpy as np

    from hoi.metrics import TC
    from hoi.plot import plot_landscape

    import matplotlib.pyplot as plt

    plt.style.use("ggplot")

    x = np.random.rand(500, 10)

    model = TC(x)
    hoi = model.fit(minsize=3, maxsize=None, method="gcmi")

    plot_landscape(hoi, model=model, plt_kwargs=dict(cmap="turbo"))
    plt.tight_layout()
    plt.show()
