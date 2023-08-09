import pytest

import numpy as np

from hoi.metrics import Oinfo, InfoTopo
from hoi.utils import get_nbest_mult


np.random.seed(0)


N_SAMPLES = 50
N_FEATURES = 6
N_VARIABLES = 5
x_2d = np.random.rand(N_SAMPLES, N_FEATURES)
x_3d = np.random.rand(N_SAMPLES, N_FEATURES, N_VARIABLES)
y_1d = np.random.rand(N_SAMPLES)
multiplets = [(0, 1), (2, 3), (0, 3, 4)]

# TARGET-FREE
# redundancy (0, 1, 2): 0->1; 0->2
# synergy    (3, 4, 5): 5 = 3 + 4
x_2d[:, 1] += x_2d[:, 0]
x_2d[:, 2] += x_2d[:, 0]
x_2d[:, 5] = x_2d[:, 3] + x_2d[:, 4]

# TARGET-RELATED
# redundancy (0, 1, y): y->0; 0->1
# synergy    (3, 4, y): 3 = 4 + y
x_3d[:, 0, 0] += y_1d
x_3d[:, 1, 0] += y_1d
x_3d[:, 3, 0] = x_3d[:, 4, 0] + y_1d


class TestMetrics(object):
    @staticmethod
    def _get_mult_idx(model, mult):
        n_cols = model.multiplets.shape[1]
        mult = np.r_[mult, [-1] * (n_cols - len(mult))].reshape(1, -1)
        return np.where((model.multiplets == mult).all(1))[0]

    @pytest.mark.parametrize("multiplets", [None, multiplets])
    @pytest.mark.parametrize("y", [None, y_1d])
    @pytest.mark.parametrize("x", [x_2d, x_3d])
    @pytest.mark.parametrize("metric", [Oinfo, InfoTopo])
    def test_definition(self, metric, x, y, multiplets):
        model = metric(x, y=y, multiplets=multiplets)
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
    @pytest.mark.parametrize("metric", [Oinfo, InfoTopo])
    def test_multiplets(self, metric, x, mult):
        # compute all oinfo
        model = metric(x.copy())
        hoi = model.fit()

        # compute oinfo for the multiplets
        model_f = metric(x.copy(), multiplets=mult)
        hoi_f = model_f.fit()

        for n_m, m in enumerate(mult):
            idx = self._get_mult_idx(model, m)
            np.testing.assert_almost_equal(hoi[idx, :], hoi_f[[n_m], :])
            np.testing.assert_array_equal(model_f.multiplets[n_m, 0 : len(m)], m)

    @pytest.mark.parametrize("y", [y_1d])
    @pytest.mark.parametrize("x", [x_2d, x_3d])
    @pytest.mark.parametrize("metric", [Oinfo, InfoTopo])
    def test_order(self, metric, x, y):
        model_tf = metric(x.copy())
        hoi_tf = model_tf.fit(minsize=2, maxsize=5)
        model_tr = metric(x.copy(), y=y)
        hoi_tr = model_tr.fit(minsize=2 + 1, maxsize=5 + 1)

        assert hoi_tr.shape == hoi_tf.shape
        assert model_tr.order.shape == model_tf.order.shape
        assert all([6 in m for m in model_tr.multiplets])
        assert all([m[o] == 6 for m, o in zip(model_tr.multiplets, model_tr.order - 1)])
        np.testing.assert_array_equal(model_tf.order + 1, model_tr.order)

    @pytest.mark.parametrize("y", [y_1d])
    @pytest.mark.parametrize("x", [x_2d, x_3d])
    @pytest.mark.parametrize("metric", [Oinfo, InfoTopo])
    def test_functional(self, metric, x, y):
        kw_best = dict(minsize=3, maxsize=3, n_best=1)

        if x.ndim == 2:
            model = metric(x.copy())
            hoi = model.fit(minsize=2, maxsize=5)

            df = get_nbest_mult(hoi, model=model, **kw_best)
            np.testing.assert_array_equal(df["multiplet"].values[0], [0, 1, 2])
            np.testing.assert_array_equal(df["multiplet"].values[1], [3, 4, 5])
        elif x.ndim == 3:
            model = metric(x.copy(), y=y)
            hoi = model.fit(minsize=2, maxsize=5)

            df = get_nbest_mult(hoi[:, 0], model=model, **kw_best)
            np.testing.assert_array_equal(df["multiplet"].values[0], [0, 1, 6])
            np.testing.assert_array_equal(df["multiplet"].values[1], [3, 4, 6])


if __name__ == "__main__":
    TestMetrics().test_functional(Oinfo, x_3d, y_1d)
