import numpy as np

from hoi.utils import normalize, landscape


def plot_landscape(
    hoi,
    model=None,
    orders=None,
    minsize=None,
    maxsize=None,
    kind="hist",
    undersampling=True,
    cbar=True,
    plot_legend=False,
    hist_kwargs={},
    plt_kwargs={},
):
    """Landscape representation of higher order interactions.

    Parameters
    ----------
    hoi : array_like
        Higher order interactions.
    model : hoi.metrics
        Higher order interaction model.
    orders: array_like, optional
        Order associated to each multiplet. The default is None.
    minsize, maxsize : int | 2, None
        Minimum and maximum size of the multiplets
    kind : {'hist', 'scatter'}
        Kind of plot. Use either:

            * 'hist' : 2D histogram of the higher order interactions
            * 'scatter' : scatter plot of the higher order interactions

    undersampling : bool | True
        If True, plot the undersampling threshold.
    cbar : bool | True
        Add colorbar.
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
    
    # get orders
    if model:
        orders = model.order

    # get computed orders
    if minsize is None:
        minsize = orders.min()
    if maxsize is None:
        maxsize = orders.max()

    # order selection
    keep = np.logical_and(orders >= minsize, orders <= maxsize)
    hoi = hoi[keep]
    orders = orders[keep]

    # switch between histogram and scatter plot
    if kind == "hist":
        hist_kwargs["output"] = "numpy"
        lscp, order, bins = landscape(hoi, orders, **hist_kwargs)

        plt.pcolormesh(order, bins, lscp, norm=LogNorm(), **plt_kwargs)
        if cbar:
            plt.colorbar(label=hist_kwargs.get("stat", "probability"))
    elif kind == "scatter" and 'c' not in plt_kwargs.keys():
        size = normalize(np.abs(hoi), to_min=2, to_max=200)
        minmax = abs(np.nanpercentile(hoi, [1, 99])).max()
        
        if 'vmin' not in plt_kwargs:
            plt_kwargs['vmin']= - minmax
        if 'vmax' not in plt_kwargs:
            plt_kwargs['vmax']= minmax    
        
        for o in range(minsize, maxsize + 1):
            keep = orders == o
            hoi_o = hoi[keep]
            plt_kwargs['c']=hoi_o
            
            x = np.random.normal(loc=o, scale=0.13, size=hoi_o.size)
            plt.scatter(
                x,
                hoi_o,
                s=size[keep],
                **plt_kwargs,
            )
    elif kind == "scatter" and 'c' in plt_kwargs.keys():
        size = normalize(np.abs(hoi), to_min=2, to_max=200)
       
        minmax = abs(np.nanpercentile(hoi, [1, 99])).max()   
        colors=plt_kwargs['c']
        labels=plt_kwargs['label']
        del plt_kwargs['c']
        del plt_kwargs['label']

        b=0
        if 'size' not in plt_kwargs.keys():
            b+=1

        else:
            siz = plt_kwargs['size']
            del plt_kwargs['size']

        for o in np.unique(colors):
            index_color=np.where(colors==o)[0]
            hoi_o = hoi[index_color]
            if b==1:
                siz=size[index_color]


            x = orders[index_color] + np.random.normal(loc=0, scale=0.08, size=hoi_o.size)
            plt.scatter(
                x,
                hoi_o,
                c=o,
                s=siz,
                label=labels[index_color][0],
                **plt_kwargs,
            )
            print(labels[index_color][0])

        if plot_legend == True:
            plt.legend()


    plt.xlabel("Order")
    plt.xticks(np.arange(minsize, maxsize + 1))
    if model:
        plt.ylabel(f"{model.__name__} [Bits]")
    else:
        plt.ylabel("HOI [Bits]")
    plt.grid(True)

    if undersampling and model:
        plt.axvline(model.undersampling, color="k", linestyle="--")

    return plt.gca()
