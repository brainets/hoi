from hoi.utils.progressbar import scan_tqdm, loop_tqdm, build_tqdm
import typing
import pytest


class TestProgressBar(object):
    @pytest.mark.parametrize("print_rate", [None, 2])
    def test_scan_tqdm(self, print_rate):
        assert isinstance(scan_tqdm(10, print_rate), typing.Callable)

    @pytest.mark.parametrize("print_rate", [None, 2])
    def loop_scan_tqdm(self, print_rate):
        assert isinstance(loop_tqdm(10, print_rate), typing.Callable)

    @pytest.mark.parametrize("print_rate", [2, 10])
    def test_build_tqdm(self, print_rate):
        val = build_tqdm(10, print_rate)
        assert isinstance(val, tuple)
        assert len(val) == 2
