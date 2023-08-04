import logging

import numpy as np

from hoi.metrics.oinfo import Oinfo
from hoi.metrics.base_hoi import HOIEstimator

logger = logging.getLogger("hoi")


class GradientOinfo(HOIEstimator):

    r"""First order Gradient O-information.

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
    x : array_like
        Standard NumPy arrays of shape (n_samples, n_features) or
        (n_samples, n_features, n_variables)
    y : array_like
        The feature of shape (n_trials,) for estimating task-related O-info
    multiplets : list | None
        List of multiplets to compute. Should be a list of multiplets, for
        example [(0, 1, 2), (2, 7, 8, 9)]. By default, all multiplets are
        going to be computed.

    References
    ----------
    Scagliarini et al., 2023 :cite:`scagliarini2023gradients`
    """

    __name__ = "Gradient O-Information"

    def __init__(self, x, y, multiplets=None, verbose=None):
        HOIEstimator.__init__(
            self, x=x, y=None, multiplets=multiplets, verbose=verbose
        )
        self._oinf_tr = Oinfo(x, y=y, multiplets=multiplets, verbose=verbose)
        self._oinf_tf = Oinfo(x, multiplets=multiplets, verbose=verbose)

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
        # ____________________________ TASK-FREE ______________________________
        hoi_tf = self._oinf_tf.fit(
            minsize=minsize, maxsize=maxsize, method=method, **kwargs
        )

        self._multiplets = self._oinf_tf._multiplets
        self._order = self._oinf_tf._order
        self._keep = self._oinf_tf._keep

        # __________________________ TASK-RELATED _____________________________
        hoi_tr = self._oinf_tr.fit(
            minsize=self._oinf_tf.minsize + 1,
            maxsize=self._oinf_tf.maxsize + 1,
            method=method,
            **kwargs
        )

        return hoi_tr - hoi_tf


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from hoi.utils import landscape, get_nbest_mult
    from matplotlib.colors import LogNorm

    plt.style.use("ggplot")

    x = np.random.rand(300, 6)
    y = x[:, 0] + x[:, 3]

    logger.setLevel("INFO")
    model = GradientOinfo(x, y=y)
    hoi = model.fit(minsize=2, maxsize=None, method="gcmi")

    print(hoi.shape)
    print(model.order.shape)
    print(model.multiplets.shape)

    print(get_nbest_mult(hoi, model=model, minsize=2, maxsize=3))

    lscp = landscape(hoi.squeeze(), model.order, output="xarray")
    lscp.plot(x="order", y="bins", cmap="jet", norm=LogNorm())
    plt.axvline(model.undersampling, linestyle="--", color="k")
    plt.grid(True)
    plt.show()
