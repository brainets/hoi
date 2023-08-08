import logging
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from hoi.core.combinatory import combinations
from hoi.core.entropies import get_entropy, prepare_for_entropy
from hoi.utils.progressbar import get_pbar


logger = logging.getLogger("hoi")


@partial(jax.jit, static_argnums=(2,))
def ent_at_index(x, idx, entropy=None):
    """Compute entropy for a specific multiplet.

    This function has to be wrapped with the entropy function.
    """
    return x, entropy(x[:, idx, :])


class HOIEstimator(object):
    def __init__(self, x, y=None, multiplets=None, verbose=None):
        # x checking
        self._x = self._prepare_data(x, y=y, multiplets=multiplets)

        if verbose not in ["INFO", "DEBUG", "ERROR"]:
            verbose = "INFO"
        logger.setLevel(verbose)

    def __iter__(self):
        """Iteration over orders."""
        for msize in range(self.minsize, self.maxsize + 1):
            yield msize

    ###########################################################################
    ###########################################################################
    #                                 I/O
    ###########################################################################
    ###########################################################################

    def _prepare_data(self, x, y=None, multiplets=None):
        """Check input x shape."""

        # force x to be 3d
        assert x.ndim >= 2
        if x.ndim == 2:
            x = x[..., np.newaxis]

        # additional variable along feature dimension
        self._task_related = isinstance(y, (list, np.ndarray, tuple))
        if self._task_related:
            y = np.asarray(y)
            if y.ndim == 1:
                assert len(y) == x.shape[0]
                y = np.tile(y.reshape(-1, 1, 1), (1, 1, x.shape[-1]))
            elif y.ndim == 2:
                assert y.shape[0] == x.shape[0]
                assert y.shape[-1] == x.shape[-1]
                y = y[:, np.newaxis, :]
            x = np.concatenate((x, y), axis=1)

        # compute only selected multiplets
        self._custom_mults = None
        if isinstance(multiplets, (list, np.ndarray)):
            self._custom_mults = [np.asarray(m) for m in multiplets]

        self.n_samples, self.n_features, self.n_variables = x.shape

        return x

    def _check_minmax(self, minsize, maxsize):
        """Define min / max size of the multiplets."""

        # check minsize / maxsize
        if not isinstance(minsize, int):
            minsize = 1
        if not isinstance(maxsize, int):
            maxsize = self.n_features
        assert isinstance(maxsize, int)
        assert isinstance(minsize, int)
        assert maxsize >= minsize
        maxsize = max(1, min(maxsize, self.n_features))
        minsize = max(1, minsize)

        self.minsize, self.maxsize = minsize, maxsize

        return minsize, maxsize

    ###########################################################################
    ###########################################################################
    #                         INFORMATION THEORY
    ###########################################################################
    ###########################################################################
    def compute_entropies(
        self, method="gcmi", minsize=1, maxsize=None, fill_value=-1, **kwargs
    ):
        """Compute entropies for all multiplets.

        Parameters
        ----------
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

        minsize : int, optional
            Minimum size of the multiplets. Default is 1.
        maxsize : int, optional
            Maximum size of the multiplets. Default is None.
        fill_value : int, optional
            Value to fill the multiplet indices with. Default is -1.
        kwargs : dict, optional
            Additional arguments to pass to the entropy function.

        Returns
        -------
        h_x : array_like
            Entropies of shape (n_mult, n_variables)
        h_idx : array_like
            Indices of the multiplets of shape (n_mult, maxsize)
        order : array_like
            Order of each multiplet of shape (n_mult,)
        """
        logger.info(f"Compute entropy with {method}")
        msg = "Entropy H(%i)"

        # ________________________________ I/O ________________________________
        # prepare the data for computing entropy
        x, kwargs = prepare_for_entropy(self._x, method, **kwargs)

        # get entropy function
        entropy = partial(
            ent_at_index, entropy=jax.vmap(get_entropy(method=method, **kwargs))
        )

        # ______________________________ ENTROPY ______________________________
        # get all of the combinations
        kw_combs = dict(maxsize=maxsize, astype="jax")
        h_idx = self.get_combinations(minsize, **kw_combs)
        order = self.get_combinations(minsize, order=True, **kw_combs)
        h_x = jnp.zeros((len(order), self.n_variables), dtype=jnp.float32)

        # get progress bar
        pbar = get_pbar(iterable=range(minsize, maxsize + 1), leave=False)

        # compute entropies
        offset = 0
        for msize in pbar:
            pbar.set_description(desc=msg % msize, refresh=False)

            # get order
            keep = order == msize
            n_mult = keep.sum()

            # compute all entropies
            _, _h_x = jax.lax.scan(entropy, x, h_idx[keep, 0:msize])

            # fill entropies
            h_x = h_x.at[offset : offset + n_mult, :].set(_h_x)

            offset += n_mult

        pbar.close()

        self._entropies = h_x
        self._multiplets = h_idx
        self._order = order

        return h_x, h_idx, order

    ###########################################################################
    ###########################################################################
    #                             COMPUTATIONS
    ###########################################################################
    ###########################################################################

    def get_combinations(self, minsize, maxsize=None, astype="jax", order=False):
        """Get combinations of features.

        Parameters
        ----------
        minsize : int
            Minimum size of the multiplets
        maxsize : int | None
            Maximum size of the multiplets. If None, minsize is used.
        astype : {'jax', 'numpy', 'iterator'}
            Specify the output type. Use either 'jax' get the data as a jax
            array [default], 'numpy' for NumPy array or 'iterator'.
        order : bool, optional
            If True, return the order of each multiplet. Default is False.

        Returns
        -------
        combinations : array_like
            Combinations of features.
        """
        return combinations(
            self.n_features, minsize, maxsize=maxsize, astype=astype, order=order
        )

    def filter_multiplets(self, mults, order):
        """Filter multiplets.

        Parameters
        ----------
        mults : array_like
            Multiplets of shape (n_mult, maxsize)
        order : array_like
            Order of each multiplet of shape (n_mult,)

        Returns
        -------
        keep : array_like
            Boolean array of shape (n_mult,) indicating which multiplets to
            keep.
        """
        # order filtering
        if self._custom_mults is None:
            keep = jnp.ones((len(order),), dtype=bool)

            if self.minsize > 1:
                logger.info(f"    Selecting order >= {self.minsize}")
                keep = jnp.where(order >= self.minsize, keep, False)

            # task related filtering
            if self._task_related:
                logger.info("    Selecting task-related multiplets")
                keep_tr = (mults == self.n_features - 1).any(1)
                keep = jnp.logical_and(keep, keep_tr)
        else:
            keep = jnp.zeros((len(order),), dtype=bool)

            for n_m, m in enumerate(self._custom_mults):
                is_order = order == len(m)
                is_mult = (mults[:, 0 : len(m)] == m).all(1)
                idx = np.where(np.logical_and(is_mult, is_order))[0]
                assert len(idx) == 1
                keep = keep.at[idx].set(True)

        self._keep = keep

        return keep

    def fit(self):  # noqa
        raise NotImplementedError()

    ###########################################################################
    ###########################################################################
    #                             PROPERTIES
    ###########################################################################
    ###########################################################################

    @property
    def entropies(self):
        """Entropies of shape (n_mult,)"""
        return np.asarray(self._entropies[self._keep])

    @property
    def multiplets(self):
        """Indices of the multiplets of shape (n_mult, maxsize).

        By convention, we used -1 to indicate that a feature has been ignored.
        """
        return np.asarray(self._multiplets[self._keep, :])

    @property
    def order(self):
        """Order of each multiplet of shape (n_mult,)."""
        return np.asarray(self._order)[np.asarray(self._keep)]

    @property
    def undersampling(self):
        """Under-sampling threshold."""
        return np.floor(np.log2(self.n_samples))
