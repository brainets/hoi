import itertools
from functools import partial
import logging

import numpy as np

import jax
import jax.numpy as jnp

from hoi.metrics.base_hoi import HOIEstimator
from hoi.utils.progressbar import scan_tqdm

logger = logging.getLogger("hoi")


###############################################################################
###############################################################################
#                     OINFO - PRECOMPUTED ENTROPIES
###############################################################################
###############################################################################


@jax.jit
def oinfo_ent(inputs, iterators):
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


###############################################################################
###############################################################################
#                     OINFO - NO PRECOMPUTED ENTROPIES
###############################################################################
###############################################################################


@partial(jax.jit, static_argnums=(2,))
def _entropy_acc(inputs, indices, entropy=None):
    x, h = inputs
    j, m_j = indices

    # compute h(x_{j})
    h_xj = entropy(x[:, [j], :])

    # compute h(x_{-j})
    h_xmj = entropy(x[:, m_j, :])

    # accumulate entropy
    h += h_xj - h_xmj

    return (x, h), None


@partial(jax.jit, static_argnums=(2, 3))
def _oinfo_no_ent(inputs, index, entropy=None, entropy_acc=None):
    data, acc = inputs
    msize = len(index)

    # tuple selection
    x_c = data[:, index, :]

    # compute h(x^{n})
    h_xn = entropy(x_c)

    # compute \sum_{j=1}^{n} (h(x_{j}) - h(x_{-j}^n))
    h_acc = jnp.zeros_like(h_xn)
    (_, h_acc), _ = jax.lax.scan(
        entropy_acc, (x_c, h_acc), (acc[:, 0], acc[:, 1:])
    )

    # compute oinfo
    oinfo = (msize - 2) * h_xn + h_acc

    return inputs, oinfo



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

    def fit(self, minsize=2, maxsize=None, method="gcmi", low_memory=True,
            **kwargs):
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
        kw = dict(minsize=minsize, maxsize=maxsize, method=method)
        if low_memory:
            return self._fit_no_ent(**kw, **kwargs)
        else:
            return self._fit_ent(**kw, **kwargs)

    def _fit_no_ent(self, minsize=2, maxsize=None, method="gcmi", **kwargs):
        """Compute Oinfo without precomputing entropies."""
        from hoi.core.entropies import get_entropy, prepare_for_entropy
        from hoi.metrics.base_hoi import ent_at_index
        from math import comb as ccomb
        from hoi.utils.progressbar import get_pbar

        # ________________________________ I/O ________________________________
        # check min and max sizes
        minsize, maxsize = self._check_minmax(minsize, maxsize)

        # prepare the data for computing entropy
        data, kwargs = prepare_for_entropy(
            self._data, method, **kwargs
        )

        # get entropy function
        entropy = jax.vmap(get_entropy(method=method, **kwargs))
        entropy_acc = partial(_entropy_acc, entropy=entropy)
        oinfo_no_ent = partial(_oinfo_no_ent, entropy=entropy,
                               entropy_acc=entropy_acc)

        # prepare output
        n_mults = sum([ccomb(self.n_features, c) for c in range(
            minsize, maxsize + 1)])
        hoi = jnp.zeros((n_mults, self.n_variables), dtype=jnp.float32)
        h_idx = jnp.full((n_mults, maxsize), -1, dtype=int)
        order = jnp.zeros((n_mults,), dtype=int)

        # get progress bar
        pbar = get_pbar(
            iterable=range(minsize, maxsize + 1), leave=False,
        )

        # ______________________________ ENTROPY ______________________________
        offset = 0
        for msize in pbar:
            pbar.set_description(desc='Oinfo (%i)' % msize, refresh=False)

            # combinations of features
            _h_idx = self.get_combinations(msize)

            # generate indices for accumulated entropies
            acc = np.mgrid[0:msize, 0:msize].sum(0) % msize

            # compute oinfo
            _, _hoi = jax.lax.scan(oinfo_no_ent, (data, acc), _h_idx)

            # fill variables
            n_combs, n_feat = _h_idx.shape
            sl = slice(offset, offset + n_combs)
            h_idx = h_idx.at[sl, 0:n_feat].set(_h_idx)
            order = order.at[sl].set(msize)
            hoi = hoi.at[sl, :].set(_hoi)

            # updates
            offset += n_combs


        self._multiplets = np.asarray(h_idx)
        self._order = np.asarray(order)
        self._keep = np.ones_like(order, dtype=bool)

        return np.asarray(hoi)


    def _fit_ent(self, minsize=2, maxsize=None, method="gcmi", **kwargs):
        """Compute Oinfo using precomputed entropies."""

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
            pbar(oinfo_ent),
            (h_x, h_idx[..., jnp.newaxis], order),
            (jnp.arange(n_mult), h_idx_2[keep], order[keep]),
        )

        return np.asarray(hoi)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from hoi.utils import landscape, digitize
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
    model = OinfoZeroLag(x)  # , y=np.random.rand(x.shape[0])                  # TASK RELATED
    hoi = model.fit(minsize=2, maxsize=None, method="gcmi", low_memory=True)

    print(hoi.shape)
    print(model.order.shape)
    print(model.multiplets.shape)
    # 0 / 0

    lscp = landscape(hoi.squeeze(), model.order, output="xarray")
    lscp.plot(x="order", y="bins", cmap="jet", norm=LogNorm())
    plt.axvline(model.undersampling, linestyle="--", color="k")
    plt.grid(True)
    plt.show()
