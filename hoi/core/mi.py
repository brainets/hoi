from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import digamma as psi

from .entropies import prepare_for_entropy

###############################################################################
###############################################################################
#                                 SWITCHER
###############################################################################
###############################################################################


def get_mi(method="gcmi", mi_type="cc", **kwargs):
    """Get Mutual-Information function.

    Parameters
    ----------
    method : {'gcmi'}
        Name of the method to compute mutual-information.
    mi_type : {'cc', 'cd'}
        Mutual-information type might depends on the type of inputs.

            * 'cc': MI between two continuous variables
            * 'cd': MI between a continuous and a discret variable

    kwargs : dict | {}
        Additional arguments sent to the mutual-information function.

    Returns
    -------
    fcn : callable
        Function to compute mutual information on variables of shapes
        (n_features, n_samples)
    """
    if method == "gcmi":
        if mi_type == "cc":
            return partial(mi_gcmi_gg, **kwargs)
        elif mi_type == "cd":
            return partial(mi_gcmi_gd, **kwargs)
    else:
        raise ValueError(f"Method {method} doesn't exist.")


###############################################################################
###############################################################################
#                             PREPROCESSING
###############################################################################
###############################################################################


def prepare_for_mi(x, y, method, **kwargs):
    """Prepare the data before computing mutual-information."""
    x, _ = prepare_for_entropy(x, method, **kwargs.copy())
    x, kwargs = prepare_for_entropy(_, method, **kwargs.copy())

    return x, y, kwargs


@partial(jax.jit, static_argnums=(2))
def compute_mi_comb(inputs, comb, mi=None):
    x, y = inputs
    x_c = x[:, comb, :]
    return inputs, mi(x_c, y)


###############################################################################
###############################################################################
#                            GAUSSIAN COPULA
###############################################################################
###############################################################################


@partial(jax.jit, static_argnums=(2, 3))
def mi_gcmi_gg(
    x: jnp.array,
    y: jnp.array,
    biascorrect: bool = True,
    demean: bool = False,
) -> jnp.array:
    """Multi-dimentional MI between two Gaussian variables in bits.

    This function compute the MI between two tensors of shapes
    (..., mvaxis, traxis)

    Parameters
    ----------
    x, y : array_like
        Arrays to consider for computing the Mutual Information. The two input
        variables x and y should have the same shape except on the mvaxis
        (if needed).
    biascorrect : bool | False
        Specifies whether bias correction should be applied to the estimated MI
    demean : bool | False
        Specifies whether the input data have to be demeaned
    shape_checking : bool | True
        Perform a reshape and check that x and y shapes are consistents. For
        high performances and to avoid extensive memory usage, it's better to
        already have x and y with a shape of (..., mvaxis, traxis) and to set
        this parameter to False

    Returns
    -------
    mi : array_like
        The mutual information with the same shape as x and y, without the
        mvaxis and traxis
    """
    # x.shape (..., x_mvaxis, traxis)
    # y.shape (..., y_mvaxis, traxis)
    ntrl = x.shape[-1]
    nvarx, nvary = x.shape[-2], y.shape[-2]
    nvarxy = nvarx + nvary

    # joint variable along the mvaxis
    xy = jnp.concatenate((x, y), axis=-2)
    if demean:
        xy -= xy.mean(axis=-1, keepdims=True)
    cxy = jnp.einsum("...ij, ...kj->...ik", xy, xy)
    cxy /= float(ntrl - 1.0)

    # submatrices of joint covariance
    cx = cxy[..., :nvarx, :nvarx]
    cy = cxy[..., nvarx:, nvarx:]

    # Cholesky decomposition
    chcxy = jnp.linalg.cholesky(cxy)
    chcx = jnp.linalg.cholesky(cx)
    chcy = jnp.linalg.cholesky(cy)

    # entropies in nats
    # normalizations cancel for mutual information
    hx = jnp.log(jnp.einsum("...ii->...i", chcx)).sum(-1)
    hy = jnp.log(jnp.einsum("...ii->...i", chcy)).sum(-1)
    hxy = jnp.log(jnp.einsum("...ii->...i", chcxy)).sum(-1)

    ln2 = jnp.log(2)
    if biascorrect:
        vec = jnp.arange(1, nvarxy + 1)
        psiterms = psi((ntrl - vec).astype(float) / 2.0) / 2.0
        dterm = (ln2 - jnp.log(ntrl - 1.0)) / 2.0
        hx = hx - nvarx * dterm - psiterms[:nvarx].sum()
        hy = hy - nvary * dterm - psiterms[:nvary].sum()
        hxy = hxy - nvarxy * dterm - psiterms[:nvarxy].sum()

    # MI in bits
    i = (hx + hy - hxy) / ln2
    return i


@partial(jax.jit, static_argnums=(2, 3, 4))
def mi_gcmi_gd(
    x: jnp.array,
    y: jnp.array,
    biascorrect: bool = True,
    demean: bool = False,
    size: int = None,
) -> jnp.array:
    """Multi-dimentional MI between a Gaussian and a discret variables in bits.

    This function is based on ANOVA style model comparison.

    Parameters
    ----------
    x, y : array_like
        Arrays to consider for computing the Mutual Information. The two input
        variables x and y should have the same shape except on the mvaxis
        (if needed).
    mvaxis : int | None
        Spatial location of the axis to consider if multi-variate analysis
        is needed
    traxis : int | -1
        Spatial location of the trial axis. By default the last axis is
        considered
    biascorrect : bool | False
        Specifies whether bias correction should be applied to the estimated MI
    demean : bool | False
        Specifies whether the input data have to be demeaned
    shape_checking : bool | True
        Perform a reshape and check that x and y shapes are consistents. For
        high performances and to avoid extensive memory usage, it's better to
        already have x and y with a shape of (..., mvaxis, traxis) and to set
        this parameter to False

    Returns
    -------
    mi : array_like
        The mutual information with the same shape as x and y, without the
        mvaxis and traxis
    """
    # Multi-dimentional shape checking
    assert isinstance(y, jnp.ndarray) and (y.ndim == 1)
    assert x.shape[-1] == len(y)

    # x.shape (..., x_mvaxis, traxis)
    nvarx, ntrl = x.shape[-2], x.shape[-1]
    # if size is None:
    #     y_transition = jnp.r_[1, jnp.diff(jnp.sort(y))]
    #     size = jnp.where(y_transition == 1, x=1, y=0).sum()
    yi_unique = jnp.unique(y, size=size)
    sh = x.shape[:-2]
    zm_shape = list(sh) + [len(yi_unique)]

    # joint variable along the mvaxis
    if not demean:
        x = x - x.mean(axis=-1, keepdims=True)

    # class-conditional entropies
    ntrl_y = jnp.zeros((size,), dtype=int)
    hcond = jnp.zeros(zm_shape, dtype=float)

    outs, _ = jax.lax.scan(
        _categorical_entropy,
        (x, y, ntrl_y, hcond),
        (jnp.arange(size), yi_unique),
    )

    ntrl_y, hcond = outs[-2], outs[-1]

    # class weights
    w = ntrl_y / float(ntrl)

    # unconditional entropy from unconditional Gaussian fit
    cx = jnp.einsum("...ij, ...kj->...ik", x, x) / float(ntrl - 1.0)
    chc = jnp.linalg.cholesky(cx)
    hunc = jnp.log(jnp.einsum("...ii->...i", chc)).sum(-1)

    ln2 = jnp.log(2)
    if biascorrect:
        vars = jnp.arange(1, nvarx + 1)

        psiterms = psi((ntrl - vars).astype(float) / 2.0) / 2.0
        dterm = (ln2 - jnp.log(float(ntrl - 1))) / 2.0
        hunc = hunc - nvarx * dterm - psiterms.sum()

        dterm = (ln2 - jnp.log((ntrl_y - 1).astype(float))) / 2.0
        psiterms = jnp.zeros_like(ntrl_y, dtype=float)
        outs, _ = jax.lax.scan(_accumulate_psiterms, (psiterms, ntrl_y), vars)
        psiterms = outs[0]
        hcond = hcond - nvarx * dterm - (psiterms / 2.0)

    # MI in bits
    i = (hunc - jnp.einsum("i, ...i", w, hcond)) / ln2
    return i


@jax.jit
def _categorical_entropy(inputs, num_yi):
    """Compute the entropy for each category of y."""
    # unwrap inputs
    x, y, ntrl_y, hcond = inputs
    num, yi = num_yi
    idx = y == yi

    # categorical demean
    n_yi = jnp.where(idx, x=1, y=0).sum()
    xm = jnp.where(idx, x=x, y=0.0)
    xmsum = jnp.where(idx, x=xm.sum(axis=-1, keepdims=True), y=0)
    xmean = xmsum / n_yi
    xm -= xmean

    # compute entropy
    cm = jnp.einsum("...ij, ...kj->...ik", xm, xm) / (n_yi - 1)
    chcm = jnp.linalg.cholesky(cm)
    _hcond = jnp.log(jnp.einsum("...ii->...i", chcm)).sum(-1)
    hcond = hcond.at[..., num].set(_hcond)
    ntrl_y = ntrl_y.at[num].set(n_yi)

    return (x, y, ntrl_y, hcond), _hcond


@jax.jit
def _accumulate_psiterms(inputs, vi):
    """Accumulated psiterms."""
    psiterms, ntrl_y = inputs
    idx = ntrl_y - vi
    psiterms += psi(idx.astype(float) / 2.0)
    return (psiterms, ntrl_y), psiterms
