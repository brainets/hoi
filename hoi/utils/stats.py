"""Statistics and summary statistics on HOI.
"""
import numpy as np


def landscape(
    x,
    mult_size,
    n_bins=100,
    centered=False,
    stat="probability",
    output="numpy",
):
    """Compute the landscape from HOI values.

    The landscape represents the how estimates of HOI are distributed per
    order.

    Parameters
    ----------
    x : array_like
        Array of data containing the HOI estimates of shape (n_hoi,)
    multi_size : array_like
        Size of the multiplet associated to each HOI estimates. It should be an
        array of shape (n_hoi,)
    n_bins : array_like | 100
        Number of bins to use to build the histogram at each order
    centered : bool | False
        Specify whether bin edges should be centered around zero
    stat : {'probability', 'frequency', 'count', 'density', 'percent'}
        Aggregate statistic to compute in each bin.

            * count: show the number of observations in each bin
            * frequency: show the number of observations divided by the bin
              width
            * probability or proportion: normalize such that bar heights sum to
              1
            * percent: normalize such that bar heights sum to 100
            * density: normalize such that the total area of the histogram
              equals 1
    output : {'numpy', 'pandas', 'xarray'} | None
        Output type. Use either :

            * 'numpy': in that case this function will returns the landscape,
              the unique orders (x-axis) and the central bins (y-axis)
            * 'pandas': 2-dimensional pandas dataframe
            * 'xarray': 2-dimensional dataarray

    Returns
    -------
    landcape : array_like
        Returns depend on the `output` parameter. Check to see what is returned
    """
    assert len(x) == len(mult_size)
    assert stat in [
        "probability",
        "proportion",
        "frequency",
        "count",
        "density",
        "percent",
    ]

    # get multiplet size
    mult_size = np.asarray(mult_size)
    minsize, maxsize = np.min(mult_size), np.max(mult_size)
    n_orders = maxsize - minsize + 1
    msize = np.arange(minsize, maxsize + 1)

    # define bin edges
    omin, omax = np.nanmin(x), np.nanmax(x)
    if centered:
        minmax = np.nanmax(np.abs([omin, omax]))
        o_range = (-minmax, minmax)
    else:
        o_range = (omin, omax)
    edges = np.histogram_bin_edges(x, bins=n_bins, range=o_range)
    edge_centers = (edges[0:-1] + edges[1::]) / 2.0

    # compute histogram
    lscp = np.zeros((n_bins, n_orders))
    for n_m, mult in enumerate(msize):
        idx = np.where(mult_size == mult)[0]
        _hist = np.histogram(x[idx], bins=edges, density=stat == "density")[0]

        if stat in ["probability", "proportion"]:
            _hist = _hist.astype(float) / np.sum(_hist)
        elif stat == "percent":
            _hist = 100.0 * _hist.astype(float) / np.sum(_hist)
        elif stat == "frequency":
            _hist = _hist.astype(float) / np.diff(edges)

        lscp[:, n_m] = _hist
    lscp[lscp == 0.0] = np.nan

    # output type
    if output == "numpy":
        return lscp, msize, edge_centers
    elif output == "pandas":
        import pandas as pd

        return pd.DataFrame(
            np.flipud(lscp), columns=msize, index=edge_centers[::-1]
        )
    elif output == "xarray":
        import xarray as xr

        attrs = dict(stat=stat, n_bins=n_bins)
        lscp = xr.DataArray(
            lscp,
            dims=("bins", "order"),
            coords=(edge_centers, msize),
            name=stat,
            attrs=attrs,
        )
        lscp.bins.attrs["unit"] = "bits"
        return lscp

    return lscp


def digitize_1d(x, n_bins):
    """One dimensional digitization."""
    assert x.ndim == 1
    x_min, x_max = x.min(), x.max()
    dx = (x_max - x_min) / n_bins
    x_binned = ((x - x_min) / dx).astype(int)
    x_binned = np.minimum(x_binned, n_bins - 1)
    return x_binned.astype(int)


def digitize_sklearn(x, **kwargs):
    """One dimensional digitization."""
    assert x.ndim == 1
    from sklearn.preprocessing import KBinsDiscretizer

    return (
        KBinsDiscretizer(**kwargs)
        .fit_transform(x.reshape(-1, 1))
        .astype(int)
        .squeeze()
    )


def digitize(x, n_bins, axis=0, use_sklearn=False, **kwargs):
    """Discretize a continuous variable.

    Parameters
    ----------
    x : array_like
        Array to discretize
    n_bins : int
        Number of bins
    axis : int | 0
        Axis along which to perform the discretization. By default,
        discretization is performed along the first axis (n_samples,)
    use_sklearn : bool | False
        If True, use sklearn.preprocessing.KBinsDiscretizer to discretize the
        data.
    kwargs : dict | {}
        Additional arguments are passed to
        sklearn.preprocessing.KBinsDiscretizer. For example, use
        `strategy='quantile'` for equal population binning.

    Returns
    -------
    x_binned : array_like
        Digitized array with the same shape as x
    """
    if not use_sklearn:
        return np.apply_along_axis(digitize_1d, axis, x, n_bins)
    else:
        kwargs["n_bins"] = n_bins
        kwargs["encode"] = "ordinal"
        kwargs["subsample"] = None
        return np.apply_along_axis(digitize_sklearn, axis, x, **kwargs)


def normalize(x, to_min=0.0, to_max=1.0):
    """Normalize the array x between to_min and to_max.

    Parameters
    ----------
    x : array_like
        The array to normalize
    to_min : int/float | 0.
        Minimum of returned array
    to_max : int/float | 1.
        Maximum of returned array

    Returns
    -------
    xn : array_like
        The normalized array
    """
    # find minimum and maximum
    if to_min is None:
        to_min = np.nanmin(x)  # noqa
    if to_max is None:
        to_max = np.nanmax(x)  # noqa

    # normalize
    if x.size:
        xm, xh = np.nanmin(x), np.nanmax(x)
        if xm != xh:
            x_n = to_max - (((to_max - to_min) * (xh - x)) / (xh - xm))
        else:
            x_n = x * to_max / xh
    else:
        x_n = x

    return x_n


def get_nbest_mult(
    hoi,
    model=None,
    orders=None,
    multiplets=None,
    n_best=5,
    minsize=None,
    maxsize=None,
    names=None,
):
    """Get the n best multiplets.

    This function requires pandas to be installed.

    Parameters
    ----------
    hoi : array_like
        Array of higher-order information.
    model : hoi.metrics, optional
        Model used to compute the higher-order information. The default is
        None.
    orders: array_like, optional
        Order associated to each multiplet. The default is None.
    multiplets: array_like, optional
        Combination of features. The default is None.
    n_best : int, optional
        Number of best multiplets to return. The default is 5.
    minsize : int, optional
        Minimum size of the multiplets. The default is None.
    maxsize : int, optional
        Maximum size of the multiplets. The default is None.
    names : list, optional
        List of names of the variables. The default is None.

    Returns
    -------
    df_best : pandas.DataFrame
        Dataframe containing the n best multiplets.
    """
    import pandas as pd

    hoi = np.asarray(hoi).squeeze()

    # get order and multiplets
    if model:
        orders = model.order
        multiplets = model.multiplets

    # get computed orders
    if minsize is None:
        minsize = orders.min()
    if maxsize is None:
        maxsize = orders.max()

    # get the data at selected order
    indices = np.arange(hoi.shape[0])
    keep_order = np.logical_and(orders >= minsize, orders <= maxsize)

    # merge into a dataframe
    df = pd.DataFrame(
        {
            "hoi": hoi[keep_order],
            "index": indices[keep_order],
            "order": orders[keep_order],
        }
    ).sort_values(by="hoi", ascending=False)

    # df selection
    is_syn = df["hoi"] < 0
    df_syn = df.loc[is_syn]
    df_syn = df_syn.iloc[-min(n_best, len(df_syn)) : :]
    df_red = df.loc[~is_syn]
    df_red = df_red.iloc[: min(n_best, len(df_red))]
    df_best = pd.concat((df_red, df_syn)).reset_index(drop=True)

    # reorder columns
    df_best = df_best[["index", "order", "hoi"]]

    # multiplets selection
    mults = []
    for m in multiplets[df_best["index"].values, :]:
        mults.append(m[m != -1])
    df_best["multiplet"] = mults

    # find names
    if names is not None:
        cols = np.asarray(names)
        names = []
        for c in df_best["multiplet"].values:
            c = np.asarray(c)
            names.append(" / ".join(cols[c].tolist()))
        df_best["names"] = names

    return df_best
