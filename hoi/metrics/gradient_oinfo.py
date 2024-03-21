import numpy as np

from hoi.metrics.oinfo import Oinfo
from hoi.metrics.base_hoi import HOIEstimator


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
        The feature of shape (n_samples,) for estimating task-related O-info
    multiplets : list | None
        List of multiplets to compute. Should be a list of multiplets, for
        example [(0, 1, 2), (2, 7, 8, 9)]. By default, all multiplets are
        going to be computed.

    References
    ----------
    Scagliarini et al., 2023 :cite:`scagliarini2023gradients`
    """

    __name__ = "Gradient O-Information"
    _encoding = True
    _positive = "redundancy"
    _negative = "synergy"
    _symmetric = True

    def __init__(self, x, y, multiplets=None, base_model=Oinfo, verbose=None):
        kw_oinfo = dict(multiplets=multiplets, verbose=verbose)
        HOIEstimator.__init__(self, x=x, y=None, **kw_oinfo)
        self._oinf_tr = base_model(x, y=y, **kw_oinfo)
        self._oinf_tf = base_model(x, **kw_oinfo)
        self.__name__ = self.__name__ + "(%s)" % base_model.__name__

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

        Returns
        -------
        hoi : array_like
            The NumPy array containing values of higher-rder interactions of
            shape (n_multiplets, n_variables)
        """
        kw_oinfo = dict(
            minsize=minsize, maxsize=maxsize, method=method, **kwargs
        )

        # ____________________________ TASK-FREE ______________________________
        hoi_tf = self._oinf_tf.fit(**kw_oinfo)

        self._multiplets = self._oinf_tf._multiplets

        # __________________________ TASK-RELATED _____________________________
        hoi_tr = self._oinf_tr.fit(**kw_oinfo)

        return hoi_tr - hoi_tf


if __name__ == "__main__":
    from hoi.utils import get_nbest_mult

    np.random.seed(0)

    x = np.random.rand(200, 7)
    y_red = np.random.rand(x.shape[0])

    # redundancy: (1, 2, 6) + (7, 8)
    x[:, 1] += y_red
    x[:, 2] += y_red
    x[:, 6] += y_red
    # synergy:    (0, 3, 5) + (7, 8)
    y_syn = x[:, 0] + x[:, 3] + x[:, 5]
    # bivariate target
    y = np.c_[y_red, y_syn]

    model = GradientOinfo(x, y=y)
    hoi = model.fit(minsize=2, maxsize=None, method="gcmi")

    print(get_nbest_mult(hoi, model=model, minsize=3, maxsize=3, n_best=3))
