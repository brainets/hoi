from tqdm.auto import tqdm
import logging

import numpy as np

from hoi.core.combinatory import combinations
from hoi.core.entropies import get_entropy, prepare_for_entropy


logger = logging.getLogger('hoi')


class HOIEstimator(object):

    def __init__(self):
        pass

    def __iter__(self):
        """Iteration over orders with progressbar."""
        pbar = self._get_pbar(
            iterable=range(self.minsize, self.maxsize + 1),
            desc=f"Order {self.minsize}"
        )

        for msize in pbar:
            pbar.set_description(desc=f"Order {msize}", refresh=False)
            yield msize

    ###########################################################################
    ###########################################################################
    #                                 I/O
    ###########################################################################
    ###########################################################################

    def _prepare_data(self, data):
        """Check input data shape."""

        # force data to be 3d
        assert data.ndim >= 2
        if data.ndim == 2:
            data = data[..., np.newaxis]

        self.n_samples, self.n_features, self.n_variables = data.shape

        return data


    def _prepare_multiplets(self, minsize, maxsize, y=None):
        """Define min / max size of the multiplets and if it's task-related."""

        # check minsize / maxsize
        if not isinstance(maxsize, int):
            maxsize = self.n_features
        assert isinstance(maxsize, int)
        assert isinstance(minsize, int)
        assert maxsize >= minsize
        maxsize = max(1, min(maxsize, self.n_features))
        minsize = max(1, minsize)

        self.minsize = minsize
        self.maxsize = maxsize

        # check in case of task-related hoi
        self.task_related = isinstance(y, (list, np.ndarray, tuple))


    def _prepare_for_entropy(self, data, method, y=None, **kwargs):
        """Prepare data before computing entropy."""

        # prepare the data for computing entropy
        data, kwargs = prepare_for_entropy(
            data, method, y, **kwargs
        )

        # get entropy function
        entropy = get_entropy(method=method, **kwargs)

        return data, entropy

    def _get_pbar(self, **kwargs):
        """Get progress bar"""
        kwargs["disable"] = logger.getEffectiveLevel() > 20
        kwargs["mininterval"] = 0.016
        kwargs["miniters"] = 1
        kwargs["smoothing"] = 0.05
        kwargs["bar_format"] = (
            "{percentage:3.0f}%|{bar}| {desc} : {n_fmt}/{total_fmt} [{elapsed}"
            "<{remaining}, {rate_fmt:>11}{postfix}]")
        return tqdm(**kwargs)


    ###########################################################################
    ###########################################################################
    #                             COMPUTATIONS
    ###########################################################################
    ###########################################################################

    def get_combinations(self, msize, as_iterator=False, as_jax=True,
                         order=False):
        return combinations(
            self.n_features, msize, task_related=self.task_related,
            as_iterator=as_iterator, as_jax=as_jax, order=order
        )

    def fit(self):  # noqa
        raise NotImplementedError()

    @property
    def multiplets(self):
        """List of multiplets."""
        multiplets = []
        for msize in self:
            multiplets += self.get_combinations(
                msize, as_iterator=False, as_jax=False)
        return multiplets

    @property
    def order(self):
        """Order of each multiplet.."""
        order = []
        for msize in self:
            order += self.get_combinations(
                msize, as_iterator=False, as_jax=False, order=True)
        return order

    @property
    def undersampling(self):
        """Under-sampling threshold."""
        return np.floor(np.log2(self.n_samples))
