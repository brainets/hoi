import logging

import numpy as np

import jax
import jax.numpy as jnp

from hoi.metrics.base_hoi import HOIEstimator
from hoi.utils.progressbar import scan_tqdm

logger = logging.getLogger("hoi")


@jax.jit
def compute_oinfo(inputs, iterators):
    # h_x = (n_mult, n_var); h_idx = (n_mult, maxsize, 1)
    h_x, h_idx, order = inputs
    _, comb, msize = iterators

    # find all of the indices
    isum = (h_idx == comb[jnp.newaxis, :]).sum((1, 2))

    # indices for h^{n}, h^{j} and h^{-j}
    i_n = jnp.logical_and(isum == msize, order == msize)
    i_j = jnp.logical_and(isum == 1, order == 1)
    i_mj = jnp.logical_and(isum == msize - 1, order == msize - 1)

    # sum entropies only when needed
    h_n = jnp.sum(h_x, where=i_n.reshape(-1, 1), axis=0)
    h_j = jnp.sum(h_x, where=i_j.reshape(-1, 1), axis=0)
    h_mj = jnp.sum(h_x, where=i_mj.reshape(-1, 1), axis=0)

    # compute o-info
    o = (msize - 2.0) * h_n + h_j - h_mj

    return inputs, o


class OinfoZeroLag(HOIEstimator):

    r"""O-information.

    The O-information is defined as the difference between the total
    correlation (TC) minus the dual total correlation (DTC):

    .. math::

        \Omega(X^{n})  &=  TC(X^{n}) - DTC(X^{n}) \\
                       &=  (n - 2)H(X^{n}) + \sum_{j=1}^{n} [H(X_{j}) - H(
                        X_{-j}^{n})]

    .. warning::

        * :math:`\Omega(X^{n}) > 0 \Rightarrow Redundancy`
        * :math:`\Omega(X^{n}) < 0 \Rightarrow Synergy`

    Parameters
    ----------
    data : array_like
        Standard NumPy arrays of shape (n_samples, n_features) or
        (n_samples, n_features, n_variables)
    y : array_like
        The feature of shape (n_trials,) for estimating task-related O-info

    References
    ----------
    Rosas et al., 2019 :cite:`rosas2019oinfo`
    """

    __name__ = "O-Information"

    def __init__(self, data, y=None, verbose=None):
        HOIEstimator.__init__(self, data=data, y=y, verbose=verbose)

    def fit(self, minsize=2, maxsize=None, method="gcmi", **kwargs):
        """Compute the O-information.

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
        pbar = scan_tqdm(n_mult, message="Oinfo")

        # compute o-info
        h_idx_2 = jnp.where(h_idx == -1, -2, h_idx)
        _, hoi = jax.lax.scan(
            pbar(compute_oinfo),
            (h_x, h_idx[..., jnp.newaxis], order),
            (jnp.arange(n_mult), h_idx_2[keep], order[keep]),
        )

        return np.asarray(hoi)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from hoi.utils import landscape
    from matplotlib.colors import LogNorm

    plt.style.use("ggplot")

    path = "/home/etienne/Downloads/data_200_trials"
    x = np.load(path, allow_pickle=True)[..., 100]
    x_min, x_max = x.min(), x.max()
    x_amp = x_max - x_min
    x_bin = np.ceil(((x - x_min) * (3 - 1)) / (x_amp)).astype(int)

    logger.setLevel("INFO")
    # model = OinfoZeroLag(digitize(x, 3, axis=1))
    # model = OinfoZeroLag(x[..., 100])
    model = OinfoZeroLag(x, y=np.random.rand(x.shape[0]))
    hoi = model.fit(minsize=2, maxsize=None, method="gcmi")

    print(hoi.shape)
    print(model.order.shape)
    print(model.multiplets.shape)
    0 / 0

    lscp = landscape(hoi.squeeze(), model.order, output="xarray")
    lscp.plot(x="order", y="bins", cmap="jet", norm=LogNorm())
    plt.axvline(model.undersampling, linestyle="--", color="k")
    plt.grid(True)
    plt.show()
