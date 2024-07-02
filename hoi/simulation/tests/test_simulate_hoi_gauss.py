import pytest

from hoi.simulation import simulate_hoi_gauss
from hoi.metrics import Oinfo, GradientOinfo

n_samples = [100, 1000, 10000]
target = [True, False]
character = ["redundancy", "synergy"]


class TestSimGauss(object):
    @pytest.mark.parametrize("n_samples", n_samples)
    @pytest.mark.parametrize("target", target)
    @pytest.mark.parametrize("character", character)
    def test_def_simulate_hoi_gauss(self, n_samples, target, character):
        """Test definition."""
        out = simulate_hoi_gauss(
            n_samples=n_samples, target=target, triplet_character=character
        )

        if target:
            assert len(out) == 2
            assert out[0].shape == (n_samples, 3)
            assert out[1].shape == (n_samples,)
        else:
            assert out.shape == (n_samples, 3)

    @pytest.mark.parametrize("n_samples", n_samples)
    @pytest.mark.parametrize("target", target)
    @pytest.mark.parametrize("character", character)
    def test_func_simulate_hoi_gauss(self, n_samples, target, character):
        """Test functional."""
        # simulate data
        out = simulate_hoi_gauss(
            n_samples=n_samples, target=target, triplet_character=character
        )

        # define the model
        if target:
            model = GradientOinfo(out[0], out[1], verbose=False)
        else:
            model = Oinfo(out, verbose=False)

        # compute hoi
        hoi = model.fit(minsize=3, maxsize=3, method="gc")

        # test shape and character
        assert hoi.shape == (1, 1)
        if character == "redundancy":
            assert hoi.squeeze() > 0
        elif character == "synergy":
            assert hoi.squeeze() < 0
