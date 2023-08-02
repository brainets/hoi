import logging

import numpy as np

import jax
import jax.numpy as jnp

from hoi.metrics.base_hoi import HOIEstimator
from hoi.utils.progressbar import scan_tqdm

logger = logging.getLogger("hoi")


###############################################################################
###############################################################################
#                            MUTUAL INFORMATION
###############################################################################
###############################################################################


@jax.jit
def find_indices(inputs, c):
    combs, keep = inputs

    keep = jnp.add(keep, (combs == c).any(1).astype(int))

    return (combs, keep), None


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
    data : array_like
        Standard NumPy arrays of shape (n_samples, n_features) or
        (n_samples, n_features, n_variables)
    y : array_like
        The feature of shape (n_trials,) for estimating task-related O-info.


    References
    ----------
    Baudot et al., 2019 :cite:`baudot2019infotopo`; Tapia et al., 2018
    :cite:`tapia2018neurotransmitter`
    """

    __name__ = "Topological Information"

    def __init__(self, data, y=None, verbose=None):
        HOIEstimator.__init__(self, data, y=y, verbose=verbose)

    def fit(self, minsize=1, maxsize=None, method="gcmi", **kwargs):
        """Compute Topological Information.

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

        Returns
        -------
        oinfo : array_like
            The O-info array of shape
            (n_multiplets, n_variables)
            where positive values reflect redundant dominated
            interactions and negative values
            stand for synergistic dominated interactions.
        """
        # ____________________________ ENTROPIES ______________________________

        minsize, maxsize = self._check_minmax(minsize, maxsize)
        h_x, h_idx, order = self.compute_entropies(
            minsize=1, maxsize=maxsize, method=method, **kwargs
        )
        n_mult = h_x.shape[0]

        # _______________________________ HOI _________________________________

        # compute order and multiply entropies
        h_x_sgn = jnp.multiply(((-1.0) ** (order.reshape(-1, 1) - 1)), h_x)
        h_idx_2 = jnp.where(h_idx == -1, -2, h_idx)

        # subselection of multiplets
        keep = self.filter_multiplets(h_idx, order)
        n_mult = keep.sum()

        # progress-bar definition
        pbar = scan_tqdm(n_mult, message="Mutual information")

        # compute mi
        _, hoi = jax.lax.scan(
            pbar(compute_mi),
            (h_idx[..., jnp.newaxis], h_x_sgn, order),
            (jnp.arange(n_mult), h_idx_2[keep]),
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
    # model = InfoTopo(digitize(x[..., 100], 3, axis=1))
    # model = InfoTopo(x[..., 100])
    model = InfoTopo(x_bin)
    hoi = model.fit(maxsize=None, method="binning")
    # 0/0

    lscp = landscape(hoi.squeeze(), model.order, output="xarray")
    lscp.plot(x="order", y="bins", cmap="jet", norm=LogNorm())
    plt.axvline(model.undersampling, linestyle="--", color="k")
    plt.show()
