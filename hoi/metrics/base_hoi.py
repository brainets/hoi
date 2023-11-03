from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from hoi.core.combinatory import combinations
from hoi.core.entropies import get_entropy, prepare_for_entropy
from hoi.utils.progressbar import get_pbar
from hoi.utils.logging import logger, set_log_level


@partial(jax.jit, static_argnums=(2,))
def ent_at_index(x, idx, entropy=None):
    """Compute entropy for a specific multiplet.

    This function has to be wrapped with the entropy function.
    """
    return x, entropy(x[:, idx, :])


class HOIEstimator(object):
    def __init__(self, x, y=None, multiplets=None, verbose=None):
        set_log_level(verbose)

        # check types of default properties
        _prop_lst = ["null", "redundancy", "synergy", "info"]
        assert isinstance(self.__name__, str)
        assert isinstance(self._encoding, bool)
        assert self._positive in _prop_lst
        assert self._negative in _prop_lst
        assert isinstance(self._symmetric, bool)

        # prepare the data
        self._x = self._prepare_data(x, y=y, multiplets=multiplets)

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

        logger.debug("    Prepare the data")

        # force x to be 3d
        assert x.ndim >= 2
        if x.ndim == 2:
            x = x[..., np.newaxis]
        self._n_features_x = x.shape[1]
        self._n_features_y = 0

        # additional variable along feature dimension
        self._has_target = isinstance(y, (list, np.ndarray, tuple))
        if self._has_target:
            x = self._merge_xy(x, y=y)

        # compute only selected multiplets
        self._custom_mults = None
        if isinstance(multiplets, (list, np.ndarray)):
            self._custom_mults = multiplets

        self.n_samples, self.n_features, self.n_variables = x.shape

        return x

    def _merge_xy(self, x, y=None):
        """Merge x and y variables."""
        assert x.ndim == 3
        n_samples, n_features, n_variables = x.shape

        if y is None:
            return y

        y = np.asarray(y)

        # trial checking
        if y.shape[0] != n_samples:
            raise IOError(
                f"The numer of sample of the variable y ({y.shape[0]}) should "
                f"match the number of sample of the variable x ({n_samples})"
            )

        # 1d / 2d / 3d merging
        if y.ndim == 1:
            y = y[:, np.newaxis]
        if y.ndim == 2:
            y = np.tile(y[:, :, np.newaxis], n_variables)
        if y.shape[2] != n_variables:
            raise IOError(
                f"The numer of variables of the variable y ({y.shape[2]}) "
                f"should match the number of variables of the variable x"
                f" ({n_variables})"
            )
        self._n_features_x = x.shape[1]
        self._n_features_y = y.shape[1]

        return np.concatenate((x, y), axis=1)

    def _split_xy(self, xy):
        """Split back the xy variables into x and y."""
        return (
            xy[:, 0 : self._n_features_x, :],
            xy[:, -self._n_features_y : :, :],
        )

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
            ent_at_index,
            entropy=jax.vmap(get_entropy(method=method, **kwargs)),
        )

        # ______________________________ ENTROPY ______________________________
        # get all of the combinations
        h_idx, order = self.get_combinations(minsize, maxsize=maxsize)

        # get progress bar
        pbar = get_pbar(iterable=range(minsize, maxsize + 1), leave=False)

        # compute entropies
        offset = 0
        h_x = jnp.zeros((len(order), self.n_variables), dtype=jnp.float32)
        for msize in pbar:
            pbar.set_description(desc=msg % msize, refresh=False)

            # get the number of features when considering y
            n_feat_xy = msize + self._n_features_y

            # combinations of features
            _h_idx = h_idx[order == msize, 0:n_feat_xy]

            # compute all entropies
            _, _h_x = jax.lax.scan(entropy, x, _h_idx)

            # fill entropies
            n_combs = _h_idx.shape[0]
            h_x = h_x.at[offset : offset + n_combs, :].set(_h_x)

            offset += n_combs

        pbar.close()

        self._entropies = h_x

        return h_x, h_idx, order

    ###########################################################################
    ###########################################################################
    #                             COMPUTATIONS
    ###########################################################################
    ###########################################################################

    def get_combinations(self, minsize, maxsize=None, astype="jax"):
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

        Returns
        -------
        combinations : array_like
            Combinations of features.
        """
        logger.info("Get list of multiplets")

        # custom list of multiplets don't require to full list of multiplets
        if self._custom_mults is not None:
            logger.info("    Selecting custom multiplets")
            _orders = [len(m) for m in self._custom_mults]
            mults = jnp.full((len(self._custom_mults), max(_orders)), -1)
            for n_m, m in enumerate(self._custom_mults):
                mults = mults.at[n_m, 0 : len(m)].set(m)
            self._multiplets = mults
        else:
            # specify whether to include target or not
            n = self._n_features_x
            if not self._has_target:
                target = None
            else:
                target = (np.arange(self._n_features_y) + n).tolist()
            if self._encoding:
                target = None

            # get the full list of multiplets
            self._multiplets = combinations(
                n,
                minsize,
                maxsize=maxsize,
                target=target,
                astype=astype,
                order=False,
                fill_value=-1,
            )

        return self._multiplets, self.order

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
        return np.asarray(self._entropies)

    @property
    def multiplets(self):
        """Indices of the multiplets of shape (n_mult, maxsize).

        By convention, we used -1 to indicate that a feature has been ignored.
        """
        return np.asarray(self._multiplets)

    @property
    def order(self):
        """Order of each multiplet of shape (n_mult,)."""
        order = (self.multiplets >= 0).sum(1)
        if self._encoding:
            return order
        else:
            return order - self._n_features_y

    @property
    def undersampling(self):
        """Under-sampling threshold."""
        return np.floor(np.log2(self.n_samples))
