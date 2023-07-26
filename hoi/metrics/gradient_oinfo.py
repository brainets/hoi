import itertools
from functools import partial
import logging

import numpy as np

import jax
import jax.numpy as jnp

from hoi.metrics.base_hoi import HOIEstimator
from hoi.metrics.oinfo_zerolag import compute_oinfo
from hoi.utils.progressbar import scan_tqdm

logger = logging.getLogger("hoi")


@jax.jit
def compute_goinfo(inputs, iterators):

    # compute \Omega({X^{n}, y})
    _, o_xy = compute_oinfo(inputs, iterators)

    # compute \Omega(X^{n})
    _, comb_xy, msize = iterators
    comb_x = jnp.where(comb_xy == comb_xy.max(), -2, comb_xy)
    _, o_x = compute_oinfo(inputs, (_, comb_x, msize - 1))

    return inputs, o_xy - o_x


class GradientOinfo(HOIEstimator):

    r"""Gradient O-information.

    The Gradient O-information is defined as the difference between the
    O-information with the target variable minus the O-information without the
    target variable :

    .. math::

        \partial_{i}\Omega(X^{n}) &= \Omega(X^{n}) - \Omega(X^{n}_{-i}) \\
                                  &= (2 - n)I(X_{i}; X^{n}_{-i}) + \sum_{
                                   k=1, k\neq i}^{n} I(X_{k}; X^{n}_{-ik})

    .. warning::

        * :math:`\partial_{i}\Omega(X^{n}) > 0 \Rightarrow Redundancy`
        * :math:`\partial_{i}\Omega(X^{n}) < 0 \Rightarrow Synergy`

    Parameters
    ----------
    data : array_like
        Standard NumPy arrays of shape (n_samples, n_features) or
        (n_samples, n_features, n_variables)
    y : array_like
        The feature of shape (n_trials,) for estimating task-related O-info

    References
    ----------
    Scagliarini et al., 2023 :cite:`scagliarini2023gradients`
    """

    __name__ = "Gradient O-Information"

    def __init__(self, data, y, verbose=None):
        HOIEstimator.__init__(self, data=data, y=y, verbose=verbose)

    def fit(self, minsize=2, maxsize=None, method="gcmi", **kwargs):
        """Compute the Gradient O-information.

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
        # ____________________________ ENTROPIES ______________________________

        minsize, maxsize = self._check_minmax(minsize, maxsize)
        h_x, h_idx, order = self.compute_entropies(
            minsize=1, maxsize=maxsize, method=method, **kwargs
        )
        assert jnp.isfinite(h_x).all()
        assert not jnp.isnan(h_x).any()

        # _______________________________ HOI _________________________________

        # subselection of multiplets
        keep = self.filter_multiplets(h_idx, order)
        n_mult = keep.sum()

        # progress-bar definition
        pbar = scan_tqdm(n_mult, message="GOinfo")

        # compute \Omega({X^{n}, y})
        h_idx_2 = jnp.where(h_idx == -1, -2, h_idx)
        _, hoi = jax.lax.scan(
            pbar(compute_goinfo),
            (h_x, h_idx[..., jnp.newaxis], order),
            (jnp.arange(n_mult), h_idx_2[keep], order[keep])
        )

        return np.asarray(hoi)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from hoi.utils import landscape, digitize, get_nbest_mult
    from matplotlib.colors import LogNorm

    plt.style.use("ggplot")

    x = np.random.rand(300, 12)
    y = x[:, 0] + x[:, 3]

    logger.setLevel("INFO")
    model = GradientOinfo(x, y=y)
    hoi = model.fit(minsize=2, maxsize=None, method="gcmi")

    print(hoi.shape)
    print(model.order.shape)
    print(model.multiplets.shape)

    print(get_nbest_mult(hoi, model, minsize=3, maxsize=3))

    lscp = landscape(hoi.squeeze(), model.order, output="xarray")
    lscp.plot(x="order", y="bins", cmap="jet", norm=LogNorm())
    plt.axvline(model.undersampling, linestyle="--", color="k")
    plt.grid(True)
    plt.show()
