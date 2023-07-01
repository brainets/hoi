"""Statistics and summary statistics on HOI.
"""
import numpy as np


def landscape(x, mult_size, n_bins=100, centered=False, stat='probability',
              output='numpy'):
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
        'probability', 'proportion', 'frequency', 'count', 'density',
        'percent']

    # get multiplet size
    mult_size = np.asarray(mult_size)
    minsize, maxsize = np.min(mult_size), np.max(mult_size)
    n_orders = maxsize - minsize + 1
    msize = np.arange(minsize, maxsize + 1)

    # define bin edges
    omin, omax = np.min(x), np.max(x)
    if centered:
        minmax = np.max(np.abs([omin, omax]))
        o_range = (-minmax, minmax)
    else:
        o_range = (omin, omax)
    edges = np.histogram_bin_edges(x, bins=n_bins, range=o_range)
    edge_centers = (edges[0:-1] + edges[1::]) / 2.

    # compute histogram
    lscp = np.zeros((n_bins, n_orders))
    for n_m, mult in enumerate(msize):
        idx = np.where(mult_size == mult)[0]
        _hist = np.histogram(
            x[idx], bins=edges, density=stat == 'density')[0]

        if stat in ['probability', 'proportion']:
            _hist = _hist.astype(float) / np.sum(_hist)
        elif stat == 'percent':
            _hist = 100. * _hist.astype(float) / np.sum(_hist)
        elif stat == 'frequency':
            _hist = _hist.astype(float) / np.diff(edges)

        lscp[:, n_m] = _hist
    lscp[lscp == 0.] = np.nan

    # output type
    if output == 'numpy':
        return lscp, msize, edge_centers
    elif output == 'pandas':
        import pandas as pd
        return pd.DataFrame(np.flipud(lscp), columns=msize,
                            index=edge_centers[::-1])
    elif output == 'xarray':
        import xarray as xr
        attrs = dict(stat=stat, n_bins=n_bins)
        lscp = xr.DataArray(
            lscp, dims=('bins', 'order'), coords=(edge_centers, msize),
            name='Landscape', attrs=attrs
        )
        lscp.bins.attrs['unit'] = 'bits'
        return lscp

    return lscp


def digitize_1d(x, n_bins):
    """One dimensional digitization."""
    x_min, x_max = x.min(), x.max()
    dx = (x_max - x_min) / n_bins
    x_binned = ((x - x_min) / dx).astype(int)
    x_binned = np.minimum(x_binned, n_bins - 1)
    return x_binned.astype(int)


def digitize(x, n_bins, axis=0):
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

    Returns
    -------
    x_binned : array_like
        Digitized array with the same shape as x
    """
    return np.apply_along_axis(digitize_1d, axis, x, n_bins)
