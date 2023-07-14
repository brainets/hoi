import numpy as np

from hoi.utils import normalize, landscape


def plot_landscape(hoi, model, minsize=None, maxsize=None, kind='hist',
                   undersampling=True, hist_kwargs={}, plt_kwargs={}):
    """Landscape representation of higher order interactions.

    Parameters
    ----------
    hoi : array_like
        Higher order interactions.
    model : hoi.metrics
        Higher order interaction model.
    minsize, maxsize : int | 2, None
        Minimum and maximum size of the multiplets
    kind : {'hist', 'scatter'}
        Kind of plot. Use either:

        * 'hist' : 2D histogram of the higher order interactions
        * 'scatter' : scatter plot of the higher order interactions
    undersampling : bool | True
        If True, plot the undersampling threshold.
    hist_kwargs : dict | {}
        Optional arguments for the histogram.
    plt_kwargs : dict | {}
        Optional arguments for the plot. If kind is 'hist', the arguments
        are passed to `plt.pcolormesh`. If kind is 'scatter', the arguments
        are passed to `plt.scatter`.

    Returns
    -------
    ax : axis
        Current axis.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # get computed orders
    orders = model.order
    if minsize is None:
        minsize = orders.min()
    if maxsize is None:
        maxsize = orders.max()

    # order selection
    keep = np.logical_and(orders >= minsize, orders <= maxsize)
    hoi = hoi[keep]
    orders = orders[keep]

    # switch between histogram and scatter plot
    if kind == 'hist':
        hist_kwargs['output'] = 'numpy'
        lscp, order, bins = landscape(hoi, orders, **hist_kwargs)

        plt.pcolormesh(
            order, bins, lscp, norm=LogNorm(), **plt_kwargs)
        plt.colorbar(label=hist_kwargs.get('stat', 'probability'))
    elif kind == 'scatter':
        size = normalize(np.abs(hoi), to_min=2, to_max=200)
        minmax = abs(np.nanpercentile(hoi, [1, 99])).max()
        for o in range(minsize, maxsize + 1):
            keep = orders == o
            hoi_o = hoi[keep]
            x = np.random.normal(loc=o, scale=0.13, size=hoi_o.size)
            plt.scatter(x, hoi_o, c=hoi_o, vmin=-minmax, vmax=minmax,
                        s=size[keep], **plt_kwargs)

    plt.xlabel('Order')
    plt.xticks(np.arange(minsize, maxsize + 1))
    plt.ylabel(f'{model.__name__} [Bits]')
    plt.grid(True)

    if undersampling:
        plt.axvline(model.undersampling, color='k', linestyle='--')

    return plt.gca()
