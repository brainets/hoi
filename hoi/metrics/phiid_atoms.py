from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from hoi.core.entropies import prepare_for_it
from hoi.core.mi import compute_mi_comb, compute_mi_comb_phi, get_mi
from hoi.metrics.base_hoi import HOIEstimator
from hoi.utils.progressbar import get_pbar


@partial(jax.jit, static_argnums=(2, 3))
def compute_phiid_atoms(inputs, comb, mi_fcn_r=None, mi_fcn=None):
    x, y, ind, ind_red, atom = inputs

    n_var = x.shape[0]

    # select combination
    x_c = x[:, comb, :]
    y_c = y[:, comb, :]

    # compute max(I(x_{-j}; S))
    _, i_minj = jax.lax.scan(mi_fcn_r, (x_c, y_c), ind_red)

    _, i_tot = mi_fcn((x, y_c), comb)

    # compute max(I(x_{-j}; S))
    _, i_maxj_forward = jax.lax.scan(mi_fcn, (x_c, y_c), ind)
    _, i_maxj_backward = jax.lax.scan(mi_fcn, (y_c, x_c), ind)

    rtr = i_minj.min(0)
    rxyta = jnp.minimum(i_minj[0, :], i_minj[2, :])
    rxytb = jnp.minimum(i_minj[1, :], i_minj[3, :])
    rxytab = i_maxj_forward.min(0)
    rabtx = jnp.minimum(i_minj[0, :], i_minj[1, :])
    rabty = jnp.minimum(i_minj[2, :], i_minj[3, :])
    rabtxy = i_maxj_backward.min(0)
    ixta = i_minj[0, :]
    ixtb = i_minj[1, :]
    iyta = i_minj[2, :]
    iytb = i_minj[3, :]
    ixtab = i_maxj_forward[1, :]
    iytab = i_maxj_forward[0, :]
    ixyta = i_maxj_backward[1, :]
    ixytb = i_maxj_backward[0, :]
    ixytab = i_tot

    knowns_to_atoms_mat = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # rtr
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rxyta
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rxytb
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rxytab
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rabtx
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Rabty
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # Rabtxy
        [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # ixta
        [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # ixtb
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # iyta
        [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # iytb
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],  # ixyta
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # ixytb
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # ixtab
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],  # iytab
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # ixytab
    ]

    knowns_to_atoms_mat = jnp.array(knowns_to_atoms_mat)
    knowns_to_atoms_mat_multd = jnp.tile(
        knowns_to_atoms_mat[jnp.newaxis, :, :], (n_var, 1, 1)
    )

    b = jnp.concatenate(
        (
            rtr[:, jnp.newaxis],
            rxyta[:, jnp.newaxis],
            rxytb[:, jnp.newaxis],
            rxytab[:, jnp.newaxis],
            rabtx[:, jnp.newaxis],
            rabty[:, jnp.newaxis],
            rabtxy[:, jnp.newaxis],
            ixta[:, jnp.newaxis],
            ixtb[:, jnp.newaxis],
            iyta[:, jnp.newaxis],
            iytb[:, jnp.newaxis],
            ixyta[:, jnp.newaxis],
            ixytb[:, jnp.newaxis],
            ixtab[:, jnp.newaxis],
            iytab[:, jnp.newaxis],
            ixytab[:, jnp.newaxis],
        ),
        axis=1,
    )

    out_ = jnp.linalg.solve(knowns_to_atoms_mat_multd, b[..., None])[..., 0]

    out = jnp.zeros(len(out_[:, 0]))

    for i in atom:
        out += out_[:, i]

    return inputs, out


class AtomsPhiID(HOIEstimator):
    r"""Integrated Information Decomposition (phiID).

    For each couple of variable the phiID is performed,
    using the Minimum Mutual Information (MMI) approach to estimate
    the redundancy to redundancy atoms in this way:

    .. math::

        Red(X,Y) =   min \{ I(X_{t- \tau};X_t), I(X_{t-\tau};Y_t),
                            I(Y_{t-\tau}; X_t), I(Y_{t-\tau};Y_t) \}

    Important note: this function only works for multiplets of size 2.
    To compute the redundancy it is better to use the dedicated function
    :class:`hoi.metrics.RedundancyphiID`.

    Parameters
    ----------
    x : array_like
        Standard NumPy arrays of shape (n_samples, n_features) or
        (n_samples, n_features, n_variables)
    multiplets : list | None
        List of multiplets to compute. Should be a list of multiplets, for
        example [(0, 1), (2, 7)]. By default, all multiplets are
        going to be computed. Note that for this functions only multiplets of
        size 2 are allowed.

    References
    ----------
    Mediano et al, 2021 :cite:`mediano2021towards`
    """

    __name__ = "phiID MMI"
    _encoding = False
    _positive = "null"
    _negative = "null"
    _symmetric = False

    def __init__(self, x, multiplets=None, verbose=None):
        HOIEstimator.__init__(
            self, x=x, multiplets=multiplets, verbose=verbose
        )

    def fit(
        self,
        minsize=2,
        tau=1,
        direction_axis=0,
        maxsize=2,
        method="gc",
        samples=None,
        atoms=["sts"],
        matrix=False,
        **kwargs,
    ):
        r"""Integrated Information Decomposition (phiID).

        Parameters
        ----------
        minsize, maxsize : int | 2, None
            Minimum and maximum size of the multiplets. Note that for this
            functions only multiplets of size 2 are allowed.
        method : {'gc', 'binning', 'knn', 'kernel', callable}
            Name of the method to compute entropy. Use either :

                * 'gc': gaussian copula entropy [default]. See
                  :func:`hoi.core.entropy_gc`
                * 'gauss': gaussian entropy. See :func:`hoi.core.entropy_gauss`
                * 'binning': binning-based estimator of entropy. Note that to
                  use this estimator, the data have be to discretized. See
                  :func:`hoi.core.entropy_bin`
                * 'knn': k-nearest neighbor estimator. See
                  :func:`hoi.core.entropy_knn`
                * 'kernel': kernel-based estimator of entropy
                  see :func:`hoi.core.entropy_kernel`
                * A custom entropy estimator can be provided. It should be a
                  callable function written with Jax taking a single 2D input
                  of shape (n_features, n_samples) and returning a float.

        samples : np.ndarray
            List of samples to use to compute HOI. If None, all samples are
            going to be used.
        tau : int | 1
            The length of the delay to use to compute the redundancy as
            defined in the phiID.
            Default 1
        direction_axis : {0,2}
            The axis on which to consider the evolution,
            0 for the samples axis, 2 for the variables axis.
            Default 0
        atoms : list | ['sts']
            List of atoms to compute. Possible atoms are:
            - 'rtr'
            - 'rtu1'
            - 'rtu2'
            - 'rts'
            - 'u1tr'
            - 'u2tr'
            - 'str'
            - 'u1tu1'
            - 'u1tu2'
            - 'u2tu1'
            - 'u2tu2'
            - 'stu1'
            - 'stu2'
            - 'u1ts'
            - 'u2ts'
            - 'sts'
            The output of this function is going to be the sum of the selected
            atoms.
        kwargs : dict | {}
            Additional arguments are sent to each MI function

        Returns
        -------
        hoi : array_like
            The NumPy array containing values of higher-order interactions of
            shape (n_multiplets, n_variables)
        """
        # ____________________________ atoms __________________________________

        dic = {
            "rtr": 0,
            "rtu1": 1,
            "rtu2": 2,
            "rts": 3,
            "u1tr": 4,
            "u2tr": 8,
            "str": 12,
            "u1tu1": 5,
            "u1tu2": 6,
            "u2tu1": 9,
            "u2tu2": 10,
            "stu1": 13,
            "stu2": 14,
            "u1ts": 7,
            "u2ts": 9,
            "sts": 15,
        }

        atom = jnp.array([dic[i] for i in atoms])

        if maxsize > 2:
            raise ValueError(
                "For AtomsPhiID, " "only multiplets of size 2 are allowed."
            )

        # ________________________________ I/O ________________________________
        # check minsize and maxsize
        minsize, maxsize = self._check_minmax(max(minsize, 2), maxsize)

        # prepare the x for computing mi
        x, kwargs = prepare_for_it(self._x, method, samples=samples, **kwargs)

        n_var, n_f, n_sam = x.shape

        # prepare mi functions
        mi_fcn = jax.vmap(get_mi(method=method, **kwargs))
        compute_mi = partial(compute_mi_comb, mi=mi_fcn)
        compute_mi_r = partial(compute_mi_comb_phi, mi=mi_fcn)
        compute_at = partial(
            compute_phiid_atoms, mi_fcn_r=compute_mi_r, mi_fcn=compute_mi
        )

        # get multiplet indices and order
        h_idx, order = self.get_combinations(minsize, maxsize=maxsize)

        # get progress bar
        pbar = get_pbar(
            iterable=range(order.min(), order.max() + 1), leave=False
        )

        # _______________________________ HOI _________________________________

        offset = 0

        if direction_axis == 2:
            hoi = jnp.zeros(
                (len(order), self.n_variables - tau), dtype=jnp.float32
            )
        else:
            hoi = jnp.zeros((len(order), self.n_variables), dtype=jnp.float32)

        for msize in pbar:
            pbar.set_description(
                desc="SynPhiIDMMI order %s" % msize, refresh=False
            )

            # combinations of features
            _h_idx = h_idx[order == msize, 0:msize]

            # define indices for I(x_{-j}; S)
            ind = (jnp.mgrid[0:msize, 0:msize].sum(0) % msize)[:, 1:]

            dd = jnp.array(np.meshgrid(jnp.arange(msize), jnp.arange(msize))).T
            ind_red = dd.reshape(-1, 2, 1)

            if direction_axis == 0:
                x_c = x[:, :, :-tau]
                y = x[:, :, tau:]

            elif direction_axis == 2:
                x_c = x[:-tau, :, :]
                y = x[tau:, :, :]

            else:
                raise ValueError("axis can be eaither equal 0 or 2.")

            # compute hoi
            _, _hoi = jax.lax.scan(
                compute_at, (x_c, y, ind, ind_red, atom), _h_idx
            )

            # fill variables
            n_combs = _h_idx.shape[0]
            hoi = hoi.at[offset : offset + n_combs, :].set(_hoi)

            # updates
            offset += n_combs

        if matrix:
            mat = np.zeros((n_f, n_f, n_var))

            for i in range(n_var):
                c = 0

                for j in range(n_f):
                    for k in np.arange(j + 1, n_f):
                        mat[j, k, i] = np.asarray(hoi)[c, i]
                        c += 1

                mat[:, :, i] = mat[:, :, i] + mat[:, :, i].T
            return mat

        else:
            return np.asarray(hoi)
