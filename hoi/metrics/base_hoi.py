import logging
from functools import partial

from math import comb as ccomb
import numpy as np
import jax
import jax.numpy as jnp

from hoi.core.combinatory import combinations
from hoi.core.entropies import get_entropy, prepare_for_entropy
from hoi.utils.progressbar import get_pbar


logger = logging.getLogger('hoi')


@partial(jax.jit, static_argnums=(2,))
def ent_at_index(x, idx, entropy=None):
    """Compute entropy for a specific multiplet.

    This function has to be wrapped with the entropy function.
    """
    return x, entropy(x[:, idx, :])


class HOIEstimator(object):

    def __init__(self, data, y=None, verbose=None):
        # data checking
        self._data = self._prepare_data(data, y=y)

        if verbose not in ['INFO', 'DEBUG', 'ERROR']:
            verbose = 'INFO'
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

    def _prepare_data(self, data, y=None):
        """Check input data shape."""

        # force data to be 3d
        assert data.ndim >= 2
        if data.ndim == 2:
            data = data[..., np.newaxis]

        # for task-related, add behavior along spatial dimension
        self._task_related = isinstance(y, (list, np.ndarray, tuple))
        if self._task_related:
            y = np.asarray(y)
            if y.ndim == 1:
                assert len(y) == data.shape[0]
                y = np.tile(y.reshape(-1, 1, 1), (1, 1, data.shape[-1]))
            elif y.ndim == 2:
                assert y.shape[0] == data.shape[0]
                assert y.shape[-1] == data.shape[-1]
                y = y[:, np.newaxis, :]
            data = np.concatenate((data, y), axis=1)


        self.n_samples, self.n_features, self.n_variables = data.shape

        return data


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

    def compute_entropies(self, method='gcmi', minsize=1, maxsize=None,
                          fill_value=-1, **kwargs):
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
        data, kwargs = prepare_for_entropy(
            self._data, method, **kwargs
        )

        # get entropy function
        entropy = partial(
            ent_at_index, entropy=jax.vmap(
            get_entropy(method=method, **kwargs)
        ))

        # prepare output
        n_mults = sum([ccomb(self.n_features, c) for c in range(
            minsize, maxsize + 1)])
        h_x = jnp.zeros((n_mults, self.n_variables), dtype=jnp.float32)
        h_idx = jnp.full((n_mults, maxsize), fill_value, dtype=int)
        order = jnp.zeros((n_mults,), dtype=int)

        # get progress bar
        pbar = get_pbar(
            iterable=range(minsize, maxsize + 1), leave=False,
        )

        # ______________________________ ENTROPY ______________________________
        offset = 0
        for msize in pbar:
            pbar.set_description(desc=msg % msize, refresh=False)

            # combinations of features
            _h_idx = self.get_combinations(msize)
            n_combs, n_feat = _h_idx.shape
            sl = slice(offset, offset + n_combs)

            # fill indices and order
            h_idx = h_idx.at[sl, 0:n_feat].set(_h_idx)
            order = order.at[sl].set(msize)

            # compute all of the entropies at that order
            _, _h_x = jax.lax.scan(entropy, data, _h_idx)
            h_x = h_x.at[sl, :].set(_h_x)

            # updates
            offset += n_combs
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

    def get_combinations(self, msize, as_iterator=False, as_jax=True,
                         order=False):
        """Get combinations of features.

        Parameters
        ----------
        msize : int
            Size of the multiplets
        as_iterator : bool, optional
            If True, return an iterator. Default is False.
        as_jax : bool, optional
            If True, return a jax array. Default is True.
        order : bool, optional
            If True, return the order of each multiplet. Default is False.
        
        Returns
        -------
        combinations : array_like
            Combinations of features.
        """
        return combinations(
            self.n_features, msize, as_iterator=as_iterator, as_jax=as_jax,
            order=order
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
        keep = jnp.ones((len(order),), dtype=bool)

        # order filtering
        if self.minsize > 1:
            logger.info(f"    Selecting order >= {self.minsize}")
            keep = jnp.where(order >= self.minsize, keep, False)

        # task related filtering
        if self._task_related:
            logger.info("    Selecting task-related multiplets")
            keep_tr = (mults == self.n_features - 1).any(1)
            keep = jnp.logical_and(keep, keep_tr)

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
