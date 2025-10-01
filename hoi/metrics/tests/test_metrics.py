import numpy as np
import pytest

from hoi.metrics import (
    DTC,
    RSI,
    TC,
    GradientOinfo,
    InfoTopo,
    InfoTot,
    Oinfo,
    RedundancyMMI,
    RedundancyphiID,
    Sinfo,
    SynergyMMI,
    AtomsPhiID,
    DOtot,
)
from hoi.utils import get_nbest_mult

np.random.seed(0)

# ---------------------------------- SETTINGS ---------------------------------
# Number of variables
N_SAMPLES = 50
N_FEATURES_X = 6
N_FEATURES_Y = 3
N_VARIABLES = 5

# metrics settings
METRICS_NET = [
    Oinfo,
    InfoTopo,
    TC,
    DTC,
    Sinfo,
    RedundancyphiID,
    AtomsPhiID,
    DOtot,
]
METRICS_ENC = [RedundancyMMI, SynergyMMI, GradientOinfo, RSI, InfoTot]
METRICS_ALL = METRICS_NET + METRICS_ENC

x_2d = np.random.rand(N_SAMPLES, N_FEATURES_X)
x_3d = np.random.rand(N_SAMPLES, N_FEATURES_X, N_VARIABLES)
y_1d = np.random.rand(N_SAMPLES, 1)
y_2d = np.random.rand(N_SAMPLES, N_FEATURES_Y)
y_3d = np.random.rand(N_SAMPLES, N_FEATURES_Y, N_VARIABLES)
multiplets = [(0, 1), (2, 3), (0, 3, 4)]

# ---------------------------------- BEHAVIOR ---------------------------------
# redundancy (0, 1, 2): 0->1; 0->2
# synergy    (3, 4, 5): 5 = 3 + 4
x_2d[:, 1] += x_2d[:, 0]
x_2d[:, 2] += x_2d[:, 0]
x_2d[:, 5] = x_2d[:, 3] + x_2d[:, 4]

# ---------------------------------- ENCODING ---------------------------------
# redundancy (0, 1, y): y->0; 0->1
# synergy    (3, 4, y): 3 = 4 + y
x_3d[:, 0, 0] += y_1d.squeeze()
x_3d[:, 1, 0] += y_1d.squeeze()
x_3d[:, 3, 0] = x_3d[:, 4, 0] + y_1d.squeeze()

# ----------------------------------- PHIID -----------------------------------
# simulate the variable x
n_features = 4
x_phiid = np.random.rand(200, 4)

# synergy between (0, 1)
for i in range(190):
    x_phiid[i, 0] = np.sum(x_phiid[i : i + 20, 1]) + 0.1 * np.sum(
        x_phiid[i : i + 20, 0]
    )
    x_phiid[i, 1] = np.sum(x_phiid[i : i + 20, 0]) + 0.1 * np.sum(
        x_phiid[i : i + 20, 1]
    )

# redundancy between (0, 2)
x_phiid[:, 2] = x_phiid[:, 0] + np.random.rand(200) * 0.05

# ---------------------------------- Dyn Oinfo --------------------------------

# simulate the variable x
n_features = 6
x_dotot = np.random.rand(200, 6)

# synergy between (0, 1, 2)
for i in range(190):
    x_dotot[i, 0] = np.sum(x_dotot[i : i + 20, 1]) + 0.1 * np.sum(
        x_dotot[i : i + 20, 0]
    )
    x_dotot[i, 1] = np.sum(x_dotot[i : i + 20, 0]) + 0.1 * np.sum(
        x_dotot[i : i + 20, 1]
    )
    x_dotot[i, 2] = 0.1 * np.sum(x_dotot[i : i + 20, 2]) + 0.1 * np.sum(
        x_dotot[i : i + 20, 0]
    )

# redundancy between (3, 4, 5)
x_dotot[:, 3] = x_dotot[:, 4] + np.random.rand(200) * 0.05
x_dotot[:, 5] = x_dotot[:, 4] + np.random.rand(200) * 0.05
# -----------------------------------------------------------------------------


class TestMetricsSmoke(object):
    @staticmethod
    def _get_mult_idx(model, mult):
        n_cols = model.multiplets.shape[1]
        mult = np.r_[mult, [-1] * (n_cols - len(mult))].reshape(1, -1)
        return np.where((model.multiplets == mult).all(1))[0]

    @pytest.mark.parametrize("multiplets", [None, multiplets])
    @pytest.mark.parametrize("y", [None, y_1d, y_2d, y_3d])
    @pytest.mark.parametrize("x", [x_2d, x_3d])
    @pytest.mark.parametrize("metric", METRICS_ALL)
    def test_definition(self, metric, x, y, multiplets):
        # x_2d & y_3d can't exist
        if isinstance(y, np.ndarray) and (x.ndim < y.ndim):
            return None

        # y == None and target-related impossible
        if (y is None) and (metric in METRICS_ENC):
            return None

        # skip phiid when there's a target
        if (y is not None) and (metric in [RedundancyphiID, DOtot]):
            return None

        # skip infotopo if multiplets or y
        if metric == InfoTopo or metric == AtomsPhiID:
            kw_def = dict()
            if (y is not None) or (multiplets is not None):
                return None
        elif metric in [RedundancyphiID, DOtot]:
            kw_def = dict(multiplets=multiplets)
        else:
            kw_def = dict(y=y, multiplets=multiplets)

        model = metric(x, **kw_def)
        hoi = model.fit()

        if (
            isinstance(y, np.ndarray)
            and not isinstance(multiplets, list)
            and (metric not in METRICS_ENC)
        ):
            np.testing.assert_array_equal(
                model.multiplets.max(1).min(), N_FEATURES_X + y.shape[1] - 1
            )

        if isinstance(multiplets, list):
            np.testing.assert_array_equal(hoi.shape[0], len(multiplets))
            assert all(
                [
                    (k[0 : len(i)] == i).all()
                    for (k, i) in zip(model.multiplets, multiplets)
                ]
            )

    @pytest.mark.parametrize("mult", [[(0, 1)], [(0, 1), (2, 3, 4)]])
    @pytest.mark.parametrize("x", [x_2d, x_3d])
    @pytest.mark.parametrize("metric", METRICS_NET)
    def test_multiplets(self, metric, x, mult):
        # skip some metric
        if metric in [InfoTopo, AtomsPhiID, DOtot]:
            return None

        # compute all hoi
        model = metric(x.copy())
        hoi = model.fit()

        # compute hoi for the multiplets
        model_f = metric(x.copy(), multiplets=mult)
        hoi_f = model_f.fit()

        for n_m, m in enumerate(mult):
            idx = self._get_mult_idx(model, m)
            np.testing.assert_almost_equal(
                hoi[idx, :], hoi_f[[n_m], :], decimal=4
            )
            np.testing.assert_array_equal(
                model_f.multiplets[n_m, 0 : len(m)], m
            )

    @pytest.mark.parametrize("y", [y_1d, y_2d, y_3d])
    @pytest.mark.parametrize("x", [x_2d, x_3d])
    @pytest.mark.parametrize("metric", METRICS_ALL)
    def test_order(self, metric, x, y):
        # ------------------------------- CHECK -------------------------------
        # x_2d & y_3d can't exist
        if isinstance(y, np.ndarray) and (x.ndim < y.ndim):
            return None

        # y == None and target-related impossible
        if (y is None) and (metric in METRICS_ENC):
            return None

        # ------------------------------ BEHAVIOR -----------------------------
        if metric in METRICS_NET:
            # special case of InfoTopo
            if metric in [InfoTopo, RedundancyphiID, DOtot]:
                model = metric(x.copy())
                model.fit(minsize=2, maxsize=5)
                np.testing.assert_array_equal(model.order.min(), 2)
                np.testing.assert_array_equal(model.order.max(), 5)
                return None

            elif metric in [AtomsPhiID]:
                model = metric(x.copy())
                model.fit(minsize=2, maxsize=2)
                np.testing.assert_array_equal(model.order.min(), 2)
                np.testing.assert_array_equal(model.order.max(), 2)
                return None
            elif metric not in [AtomsPhiID]:
                # compute task-free and task-related
                model_tf = metric(x.copy())
                hoi_tf = model_tf.fit(minsize=2, maxsize=5)
                model_tr = metric(x.copy(), y=y.copy())
                hoi_tr = model_tr.fit(minsize=2, maxsize=5)

                # check orders
                np.testing.assert_array_equal(hoi_tr.shape, hoi_tf.shape)
                np.testing.assert_array_equal(model_tf.order, model_tr.order)
                np.testing.assert_array_equal(model_tf.order.min(), 2)
                np.testing.assert_array_equal(model_tf.order.max(), 5)
                for m, o in zip(model_tr.multiplets, model_tr.order):
                    for n_y in range(y.shape[1]):
                        assert N_FEATURES_X + n_y in m
                        np.testing.assert_array_equal(
                            m[o + n_y], N_FEATURES_X + n_y
                        )

        # ------------------------------ ENCODING -----------------------------
        if metric in METRICS_ENC:
            model = metric(x.copy(), y=y.copy())
            model.fit(minsize=2, maxsize=5)

            # check orders
            np.testing.assert_array_equal(model.order.min(), 2)
            np.testing.assert_array_equal(model.order.max(), 5)

    @pytest.mark.parametrize("y", [y_1d, y_2d, y_3d])
    @pytest.mark.parametrize("x", [x_2d, x_3d])
    @pytest.mark.parametrize("samples", [None, 20])
    @pytest.mark.parametrize("metric", METRICS_ALL)
    def test_samples(self, metric, x, y, samples):
        """Test sample selection."""
        # ------------------------------- CHECK -------------------------------
        # x_2d & y_3d can't exist
        if isinstance(y, np.ndarray) and (x.ndim < y.ndim):
            return None

        # y == None and target-related impossible
        if (y is None) and (metric in METRICS_ENC):
            return None

        # -------------------------------- HOI --------------------------------
        if metric in METRICS_NET:
            model = metric(x.copy(), verbose=False)
        elif metric in METRICS_ENC:
            model = metric(x.copy(), y=y.copy(), verbose=False)

        if metric == AtomsPhiID:
            model.fit(minsize=2, maxsize=2, samples=samples)
        else:
            model.fit(minsize=3, maxsize=3, samples=samples)


class TestMetricsFunc(object):
    @pytest.mark.parametrize("xy", [(x_2d, None), (x_3d, y_1d)])
    @pytest.mark.parametrize("metric", [Oinfo])
    def test_oinfo(self, metric, xy):
        x, y = xy
        if y is None:
            model = metric(x.copy(), y=None)
        else:
            model = metric(x.copy(), y=y.copy())
        hoi = model.fit(minsize=2, maxsize=5)

        if x.ndim == 2:
            df = get_nbest_mult(
                hoi, model=model, minsize=3, maxsize=3, n_best=1
            )
            np.testing.assert_array_equal(df["multiplet"].values[0], [0, 1, 2])
            np.testing.assert_array_equal(df["multiplet"].values[1], [3, 4, 5])
        elif (x.ndim == 3) and metric not in [InfoTopo]:
            df = get_nbest_mult(
                hoi[:, 0], model=model, minsize=2, maxsize=2, n_best=1
            )
            np.testing.assert_array_equal(df["multiplet"].values[0], [0, 1, 6])
            np.testing.assert_array_equal(df["multiplet"].values[1], [3, 4, 6])

    @pytest.mark.parametrize("xy", [(x_2d, None)])
    @pytest.mark.parametrize("metric", [InfoTopo])
    def test_infotopo(self, metric, xy):
        x, y = xy
        model = metric(x.copy())
        hoi = model.fit(minsize=2, maxsize=5)

        df = get_nbest_mult(hoi, model=model, minsize=3, maxsize=3, n_best=1)
        np.testing.assert_array_equal(df["multiplet"].values[0], [0, 1, 2])
        np.testing.assert_array_equal(df["multiplet"].values[1], [3, 4, 5])

    @pytest.mark.parametrize("xy", [(x_2d, None)])
    @pytest.mark.parametrize("metric", [TC, DTC, Sinfo])
    def test_tc_dtc_sinfo(self, metric, xy):
        x, y = xy
        model = metric(x.copy())
        hoi = model.fit(minsize=2, maxsize=5)

        df = get_nbest_mult(hoi, model=model, minsize=3, maxsize=3, n_best=1)
        np.testing.assert_array_equal(df["multiplet"].values[0], [3, 4, 5])

    @pytest.mark.parametrize("xy", [(x_3d, y_1d)])
    @pytest.mark.parametrize("metric", [RedundancyMMI])
    def test_redmmi(self, metric, xy):
        x, y = xy
        model = metric(x.copy(), y=y)
        hoi = model.fit(minsize=2, maxsize=5)

        df = get_nbest_mult(
            hoi[:, 0], model=model, minsize=2, maxsize=2, n_best=1
        )
        np.testing.assert_array_equal(df["multiplet"].values[0], [0, 1])

    @pytest.mark.parametrize("xy", [(x_3d, y_1d)])
    @pytest.mark.parametrize("metric", [SynergyMMI])
    def test_synmmi(self, metric, xy):
        x, y = xy
        model = metric(x.copy(), y=y)
        hoi = model.fit(minsize=2, maxsize=5)

        df = get_nbest_mult(
            hoi[:, 0], model=model, minsize=2, maxsize=2, n_best=1
        )
        np.testing.assert_array_equal(df["multiplet"].values[0], [3, 4])

    @pytest.mark.parametrize("xy", [(x_3d, y_1d)])
    @pytest.mark.parametrize("metric", [GradientOinfo, RSI])
    def test_goinfo_rsi(self, metric, xy):
        x, y = xy
        model = metric(x.copy(), y=y)
        hoi = model.fit(minsize=2, maxsize=5)
        if metric == RSI:
            hoi = -hoi

        df = get_nbest_mult(
            hoi[:, 0], model=model, minsize=2, maxsize=2, n_best=1
        )
        np.testing.assert_array_equal(df["multiplet"].values[0], [0, 1])
        np.testing.assert_array_equal(df["multiplet"].values[1], [3, 4])

    @pytest.mark.parametrize("xy", [(x_3d, y_1d)])
    @pytest.mark.parametrize("metric", [InfoTot])
    def test_infotot(self, metric, xy):
        x, y = xy
        model = metric(x.copy(), y=y)
        hoi = model.fit(minsize=2, maxsize=5)

        df = get_nbest_mult(
            hoi[:, 0], model=model, minsize=2, maxsize=2, n_best=2
        )
        np.testing.assert_array_equal(df["multiplet"].values[0], [3, 4])

    @pytest.mark.parametrize("xy", [(x_phiid, None)])
    @pytest.mark.parametrize("metric", [RedundancyphiID, AtomsPhiID])
    def test_phiid(self, metric, xy):

        x, y = xy
        model = metric(x.copy())
        hoi = model.fit(minsize=2, maxsize=2)

        df = get_nbest_mult(hoi, model=model, minsize=2, maxsize=2, n_best=1)

        if metric == RedundancyphiID:
            mult = [0, 2]
        elif metric == AtomsPhiID:
            mult = [0, 1]
        np.testing.assert_array_equal(df["multiplet"].values[0], mult)

    @pytest.mark.parametrize("xy", [(x_dotot, None)])
    @pytest.mark.parametrize("metric", [DOtot])
    def test_dotot(self, metric, xy):
        x, y = xy
        model = metric(x.copy())
        hoi = model.fit(minsize=3, maxsize=3)

        df = get_nbest_mult(hoi, model=model, minsize=3, maxsize=3, n_best=1)
        np.testing.assert_array_equal(df["multiplet"].values[0], [0, 1, 2])
        np.testing.assert_array_equal(df["multiplet"].values[-1], [3, 4, 5])
