"""
Quick introduction to jax
=========================

`Jax <https://github.com/google/jax>`_ is a Python library allowing to perform
linear algebra operations on CPU or GPU using the same code.

In this short example we will introduce to some basics of Jax, at least the
modules and functions used in HOI. In particular, we will see :

1. The jax.numpy module
2. How to compile functions using jit
3. How to use vmap
4. How to write efficient `for` loops in jax
"""


###############################################################################
# Jax.numpy module
# ----------------
#
# Jax has a module called `numpy`. As the name suggests, it allows to write
# NumPy like code except that it can be executed on CPU or on GPU.

# %%
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

import timeit

x = jnp.asarray([1, 2, 3, 1, 6, 9, 3])

print(f"x : {x}")
print(f"Sum : {x.sum()}")
print(f"Mean : {x.mean()}")
print(f"Min / max : {x.min()} / {x.max()}")
print(f"Unique elements : {jnp.unique(x)}")
print(f"Dot product :\n{jnp.dot(x.reshape(-1, 1), x.reshape(1, -1))}")

# %%
# One noticeable difference is how to change values in an array :

print(f"Updating value : {x.at[0].set(33)}")

# %%
# Further resources :
#
#     * https://jax.readthedocs.io/en/latest/jax.numpy.html : NumPy
#        functions implemented in jax
#     * https://jax.readthedocs.io/en/latest/jax.scipy.html : equivalent for
#       scipy

###############################################################################
# Compiling functions using jit
# -----------------------------
#
# Functions written in Jax can be compiled using `jax.jit`. This can lead to
# higher performance.


# Number of times to repeat the test
number = 100000
x_np = np.random.rand(1000)
y_np = np.random.rand(1000)
x_jax = jnp.asarray(x_np)
y_jax = jnp.asarray(y_np)


def numpy_eucl():
    return np.sqrt(np.sum((x_np - y_np) ** 2))


@jax.jit
def jax_eucl():
    return jnp.sqrt(jnp.sum((x_jax - y_jax) ** 2))


jax_eucl()  # dry run


t1 = timeit.timeit(numpy_eucl, number=number)
print(f"Time taken by pure NumPy function: {t1} seconds")

t2 = timeit.timeit(jax_eucl, number=number)
print(f"Time taken by jitted function: {t2} seconds")

# %%
# if the function takes optional argument, you can use the `static_argnums`
# and specify the position of the optional argument.


@partial(jax.jit, static_argnums=1)
def fcn(x, exponent=3):
    return x**exponent


print(fcn(x, exponent=3))

# %%
# Further resources :
#
#     * https://jax.readthedocs.io/en/latest/jit-compilation.html : jax
#       tutorial on how to use jit


###############################################################################
# vmap : vectorize a function
# ---------------------------
#
# vmap allows you to vectorize a function. In short, let say that you have a
# function that takes as an input a vector and return a floating point. Then,
# imagine that you've a 3d array and you want to apply this function along the
# first two dimensions. vmap allows you to do such thing.


def minmax(x):
    """This function returns the distance between the max and the min of a
    vector, divided by 2.
    """
    return (x.max() - x.min()) / 2.0


# define a 2d array
x = jnp.asarray(np.random.rand(10, 20))
print(x.shape)

# let's apply our function to the first and second axis
minmax_2d = jax.vmap(minmax, in_axes=0)
print(minmax_2d(x).shape)

# %%
# the function kind of loop of the first axis and for each vector of shape
# (20,), apply the minmax function. The same can be done over the second axis.

minmax_2d = jax.vmap(minmax, in_axes=1)
print(minmax_2d(x).shape)

# %%
# Now imagine that you have a 3d array x and you want to apply the function over
# the first and second axes. You can wrap your function twice with vmap.

# define a 3d array
x = jnp.asarray(np.random.rand(10, 20, 30))
print(x.shape)

# wrap twice
minmax_3d = jax.vmap(jax.vmap(minmax, in_axes=0), in_axes=0)
print(minmax_3d(x).shape)

# finally, you can jit your vectorize function
minmax_3d_jit = jax.jit(minmax_3d)
print(minmax_3d(x).shape)

# %%
# Further resources :
#
#     * https://jax.readthedocs.io/en/latest/automatic-vectorization.html : jax
#       tutorial on how to use vmap


###############################################################################
# For loops with jax
# ------------------
#
# For loops in Python are known to be relatively slow. Jax allows to have
# compiled and therefore efficient for loops. For an introduction to
# `jax.lax.scan`, see :
#
#     * https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan :
#       official documentation
#     * https://www.nelsontang.com/blog/a-friendly-introduction-to-scan-with-jax :
#       nice introduction to `jax.lax.scan`

