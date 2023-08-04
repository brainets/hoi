import pytest

import numpy as np

from hoi.metrics import Oinfo


N_SAMPLES = 50
N_FEATURES = 6
N_VARIABLES = 5
x_2d = np.random.rand(N_SAMPLES, N_FEATURES)
x_3d = np.random.rand(N_SAMPLES, N_FEATURES, N_VARIABLES)
y_2d = np.random.rand(N_SAMPLES)
multiplets = [(0, 1), (2, 3), (0, 3, 4)]


class _TestMetric(object):
    @staticmethod
    def _get_mult_idx(model, mult):
        n_cols = model.multiplets.shape[1]
        mult = np.r_[mult, [-1] * (n_cols - len(mult))].reshape(1, -1)
        return np.where((model.multiplets == mult).all(1))[0]


class TestOinfo(_TestMetric):
    @pytest.mark.parametrize("multiplets", [None, multiplets])
    @pytest.mark.parametrize("y", [None, y_2d])
    @pytest.mark.parametrize("x", [x_2d, x_3d])
    def test_definition(self, x, y, multiplets):
        model = Oinfo(x, y=y, multiplets=multiplets)
        hoi = model.fit()

        if isinstance(y, np.ndarray) and not isinstance(multiplets, list):
            assert all([N_FEATURES in k for k in model.multiplets])

        if isinstance(multiplets, list):
            assert hoi.shape[0] == len(multiplets)
            assert all(
                [
                    (k[0 : len(i)] == i).all()
                    for (k, i) in zip(model.multiplets, multiplets)
                ]
            )

    @pytest.mark.parametrize("mult", [[(0, 1)], [(0, 1), (2, 3, 4)]])
    @pytest.mark.parametrize("x", [x_2d, x_3d])
    def test_multiplets(self, x, mult):
        # compute all oinfo
        model = Oinfo(x)
        hoi = model.fit()

        # compute oinfo for the multiplets
        model_f = Oinfo(x, multiplets=mult)
        hoi_f = model_f.fit()

        for n_m, m in enumerate(mult):
            idx = self._get_mult_idx(model, m)
            np.testing.assert_array_equal(hoi[idx, :], hoi_f[[n_m], :])


if __name__ == "__main__":
    TestOinfo().test_multiplets(x_2d, [(0, 1), (2, 3, 4)])
