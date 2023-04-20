"""Dynamic and task-related higher-order interactions."""
from math import comb

from functools import partial
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

import numpy as np
import xarray as xr

import itertools

from frites.conn import conn_io
from frites.io import logger, check_attrs
from frites.core import copnorm_nd


@partial(jax.jit, static_argnums=1)
def ent_g(x: jnp.array, biascorrect=True) -> jnp.array:
    """Entropy of a tensor of shape (..., n_vars, n_trials)
    Parameters
    ----------
    x : jnp.array
        Input tensor of shape (..., n_vars, n_trials).
    biascorrect : bool, optional
        Whether to apply bias correction. Default is True.
    Returns
    -------
    jnp.array
        Entropy of the input tensor in nats which is is a measure of the 
        uncertainty or disorder of the input tensor x.
    """
    nvarx, ntrl = x.shape[-2], x.shape[-1]

    # covariance
    c = jnp.einsum("...ij, ...kj->...ik", x, x)
    c /= float(ntrl - 1.0)
    chc = jnp.linalg.cholesky(c)

    # entropy in nats
    hx = jnp.log(jnp.einsum("...ii->...i", chc)).sum(-1) + 0.5 * nvarx * (
        jnp.log(2 * jnp.pi) + 1.0
    )

    ln2 = jnp.log(2)
    if biascorrect:
        psiterms = (
            jsp.digamma((ntrl - jnp.arange(1, nvarx + 1).astype(float)) / 2.0) / 2.0
        )
        dterm = (ln2 - jnp.log(ntrl - 1.0)) / 2.0
        hx = hx - nvarx * dterm - psiterms.sum()

    return hx


@partial(jax.jit)
def oinfo_smult(x: jnp.array, ind: jnp.array) -> jnp.array:
    """Compute the O-info of a single multiplet.

    Parameters
    ----------
    x : jnp.array
        Input tensor of shape (..., n_vars, n_trials).
    ind : jnp.array
        Indices for tensor computations.

    Returns
    -------
    jnp.array
        The O-info array of shape (..., n_trials) where positive values
        reflect redundant dominated interactions and negative values stand for
        synergistic dominated interactions.
    """
    nvars = x.shape[-2]
    o = (nvars - 2) * ent_g(x) + (
        ent_g(x[..., jnp.newaxis, :]) - ent_g(x[..., ind, :])
    ).sum(1)
    return o


@partial(jax.jit)
def oinfo_mmult(x: jnp.array, comb: jnp.array) -> (jnp.array, jnp.array):
    """Compute the O-info on several multiplets.

    Parameters
    ----------
    x : jnp.array
        Input tensor of shape (..., n_vars, n_trials).
    comb : jnp.array
        Indices of the combinations for multiplets.

    Returns
    -------
    Tuple[jnp.array, jnp.array]
        This function returns the o-info about more than one multiplet by calling 
        the oinf_smult on each multiply individually. 
    """
    # build indices
    msize = len(comb)
    ind = jnp.mgrid[0:msize, 0:msize].sum(0) % msize

    return x, oinfo_smult(x[:, comb, :], ind[:, 1:])


def combinations(n, k, roi, task_related=False, sort=True):
    """Get combinations.
    
    Parameters
    ----------
    n : an integer representing the total number of elements in the set
    k : an integer representing the size of the combinations to be generated
    roi : a numpy array containing the names of the elements in the set
    task_related : a boolean flag (default is False) indicating whether 
    to add a final column to the combinations indicating the task-related 
    behavior of each element
    sort : a boolean flag (default is True) indicating whether to sort the 
    combinations lexicographically

    Returns
    -------
    combs : jnp.array 
        An array of shape (n_combinations, k) representing all possible 
        combinations of k brain regions.
    roi_st : list  
        A list of strings representing the names of each combination. If sort is True, 
        the names will be sorted alphabetically. If task_related is True, 
        the last column of each combination will represent the behavior.
    """
    combs = np.array(list(itertools.combinations(np.arange(n), k)))

    # add behavior as a final columns
    if task_related:
        combs = np.c_[combs, np.full((combs.shape[0],), n)]

    # build brain region names
    if not sort:
        roi_st = ["-".join(r) for r in roi[combs].tolist()]
    else:
        roi_st = ["-".join(r) for r in np.sort(roi[combs]).tolist()]

    return jnp.asarray(combs), roi_st


def conn_oinfo_jax(
    data, y=None, times=None, roi=None, minsize=3, maxsize=5, sort=True, verbose=None
):
    """Dynamic, possibly task-related, higher-order interactions.

    Parameters
    ----------
    data : array_like
        Electrophysiological data. Several input types are supported :

            * Standard NumPy arrays of shape (n_epochs, n_roi, n_times)
            * mne.Epochs
            * xarray.DataArray of shape (n_epochs, n_roi, n_times)

    y : array_like
        The feature of shape (n_trials,) for estimating task-related O-info.
    roi : array_like | None
        Array of region of interest name of shape (n_roi,)
    times : array_like | None
        Array of time points of shape (n_times,)
    minsize, maxsize : int | 3, 5
        Minimum and maximum size of the multiplets

    Returns
    -------
    oinfo : array_like
        The O-info array of shape (n_multiplets, n_times) where positive values
        reflect redundant dominated interactions and negative values stand for
        synergistic dominated interactions.
    """
    # ________________________________ INPUTS _________________________________
    # inputs conversion
    is_task_related = isinstance(y, (str, list, np.ndarray, tuple))
    kw_links = {"directed": False, "net": False}
    data, cfg = conn_io(
        data,
        y=y,
        times=times,
        roi=roi,
        name="DynOinfo",
        verbose=verbose,
        kw_links=kw_links,
    )

    # extract variables
    x, attrs = data.data, cfg["attrs"]
    y, roi, times = data["y"].data, data["roi"].data, data["times"].data
    n_roi = len(roi)

    # get the maximum size of the multiplets investigated
    if not isinstance(maxsize, int):
        maxsize = n_roi
    maxsize = max(1, maxsize)
    assert maxsize > minsize

    # get the number of multiplets
    n_mults = 0
    for msize in range(minsize, maxsize + 1):
        n_mults += comb(n_roi, msize)

    logger.info(
        f"Compute the {'task-related ' * is_task_related} HOI "
        f"(min={minsize}; max={maxsize})"
    )

    # ________________________________ O-INFO _________________________________
    logger.info("    Copnorm the data")

    # for task-related, add behavior along spatial dimension
    if is_task_related:
        y = np.tile(y.reshape(-1, 1, 1), (1, 1, len(times)))
        x = np.concatenate((x, y), axis=1)
        roi = np.r_[roi, ["beh"]]

    # copnorm and demean the data
    x = copnorm_nd(x.copy(), axis=0)
    x = x - x.mean(axis=0, keepdims=True)

    # make the data (n_times, n_roi, n_trials)
    x = jnp.asarray(x.transpose(2, 1, 0))

    oinfo, roi_o = [], []
    for msize in range(minsize, maxsize + 1):
        # ----------------------------- MULTIPLETS ----------------------------
        logger.info(f"    Multiplets of size {msize}")
        combs, _roi_o = combinations(
            n_roi, msize, roi, task_related=is_task_related, sort=sort
        )
        roi_o += _roi_o

        # ------------------------------- O-INFO ------------------------------
        _, _oinfo = jax.lax.scan(oinfo_mmult, x, combs)

        oinfo.append(np.asarray(_oinfo))
    oinfo = np.concatenate(oinfo, axis=0)

    # _______________________________ OUTPUTS _________________________________
    attrs.update(dict(task_related=is_task_related, minsize=minsize, maxsize=maxsize))
    oinfo = xr.DataArray(
        oinfo,
        dims=("roi", "times"),
        coords=(roi_o, times),
        name="Oinfo",
        attrs=check_attrs(attrs),
    )

    return oinfo


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from frites import set_mpl_style
    import seaborn as sns
    import time as tst

    set_mpl_style()

    ###########################################################################
    n_trials = 300
    n_roi = 4
    n_times = 600

    redundancy = [
        (0, 1, 3),
        # (2, 3, 5)
        # (1, 2, 4, 5)
    ]
    synergy = [(0, 1, 2)]
    ###########################################################################

    def set_redundancy(x, redundancy, sl, win, trials):
        for m in redundancy:
            x[:, m, sl] += 0.8 * trials.reshape(-1, 1, 1) * win
        return x

    def set_synergy(x, synergy, sl, win, trials):
        for m in synergy:
            blocks = np.array_split(np.arange(n_trials), len(m))
            for n_b, b in enumerate(blocks):
                x[b, m[n_b], sl] += trials[b].reshape(-1, 1) * win[0, ...]
        return x

    # generate the data
    x = np.random.rand(n_trials, n_roi, n_times)
    roi = np.array([f"r{r}" for r in range(n_roi)])[::-1]
    trials = np.random.rand(n_trials)
    # trials = np.arange(n_trials)
    # times = (np.arange(n_times) - 200) / 128.
    times = np.arange(n_times)
    win = np.hanning(100).reshape(1, 1, -1)

    # introduce (redundant, synergistic) information
    x = set_redundancy(x, redundancy, slice(200, 300), win, trials)
    x = set_synergy(x, synergy, slice(300, 400), win, trials)

    x = xr.DataArray(x, dims=("trials", "roi", "times"), coords=(trials, roi, times))

    start_time = tst.time()
    oinfo = conn_oinfo_jax(x, minsize=3, maxsize=4, y=None, roi="roi", times="times")
    print(oinfo)
    end_time = tst.time()
    print(f"Elapsed time : {end_time - start_time}")
    # exit()
    print(oinfo.shape)

    # print(oinfo)
    vmin, vmax = np.nanpercentile(oinfo.data, [1, 99])
    minmax = max(abs(vmin), abs(vmax))
    vmin, vmax = -minmax, minmax

    # plot the results
    df = oinfo.to_pandas()
    plt.pcolormesh(df.columns, df.index, df.values, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlabel("Times")
    plt.axvline(0.0, color="k")

    redu = ["-".join(roi[list(r)]) for r in redundancy]
    syn = ["-".join(roi[list(r)]) for r in synergy]
    for n_k, k in enumerate(oinfo["roi"].data):
        if k.replace("-beh", "") in redu:
            plt.gca().get_yticklabels()[n_k].set_color("red")
        if k.replace("-beh", "") in syn:
            plt.gca().get_yticklabels()[n_k].set_color("blue")

    plt.show()
