import typing

import pytest
from jax_tqdm import loop_tqdm, scan_tqdm
from tqdm.auto import tqdm

from hoi.utils.progressbar import get_pbar


class TestProgressBar(object):
    @pytest.mark.parametrize("print_rate", [None, 2])
    def test_scan_tqdm(self, print_rate):
        assert isinstance(scan_tqdm(10, print_rate), typing.Callable)

    @pytest.mark.parametrize("print_rate", [None, 2])
    def loop_scan_tqdm(self, print_rate):
        assert isinstance(loop_tqdm(10, print_rate), typing.Callable)

    def test_get_pbar(self):
        """Test function get_pbar."""
        assert isinstance(get_pbar(iterable=range(10), leave=False), tqdm)
