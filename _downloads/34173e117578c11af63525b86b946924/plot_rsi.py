"""
Redundancy-Synergy Index
========================

This example illustrates how to use and interpret the Redundancy-Synergy
Index (RSI).
"""
import numpy as np

from hoi.metrics import RSI
from hoi.utils import get_nbest_mult
from hoi.plot import plot_landscape

import matplotlib.pyplot as plt
plt.style.use('ggplot')


###############################################################################
# Definition
# ----------
#
# The RSI is a multivariate measure of information capable of disentangling
# whether a subset of a variable `X`` carry either redundant or synergistic
# information about a variable `Y`. The RSI is defined as :

# %%
# .. math::
#     RSI(S; Y) \equiv I(S; Y) - \sum_{x_{i}\in S} I(x_{i}; Y)
#
# with :
#
# .. math::
#     S = x_{1}, ..., x_{n}
#
# Positive values of RSI stand for synergy while negative values of RSI reflect
# redundancy between the `X` and `Y` variables.

###############################################################################
# Simulate univariate redundancy
# ------------------------------
#
# A very simple way to simulate redundancy is to observe that if a triplet of
# variables :math:`X_{1}, X_{2}, X_{3}` receive a copy of a variable :math:`Y`,
# we will observe redundancy between :math:`X_{1}, X_{2}, X_{3}` and :math:`Y`.
# For further information about how to simulate redundant and synergistic
# interactions, checkout the example
# :ref:`sphx_glr_auto_examples_tutorials_plot_sim_red_syn.py`

# lets start by simulating a variable x with 200 samples and 7 features
x = np.random.rand(200, 7)

# now we can also generate a univariate random variable y
y = np.random.rand(x.shape[0])

# we now send the variable y in the column (1, 3, 5) of x
x[:, 1] += y
x[:, 3] += y
x[:, 5] += y

# define the RSI model and launch it
model = RSI(x, y)
hoi = model.fit(minsize=3, maxsize=5)

# now we can take a look at the multiplets with the highest and lowest values
# of RSI. We will only select the multiplets of size 3 here
df = get_nbest_mult(hoi, model=model, minsize=3, maxsize=3, n_best=3)
print(df)

# %%
# as you see from the printed table, the multiplet with the lowest (i.e. the
# most redundant multiplets) is (1, 3, 5).

###############################################################################
# Simulate multivariate redundancy
# --------------------------------
#
# In the example above, we simulated a univariate :math:`Y` variable (i.e.
# single column). However, it's possible to simulate a multivariate variable.

# simulate x again
x = np.random.rand(200, 7)

# simulate a bivariate y variable
y = np.c_[np.random.rand(x.shape[0]), np.random.rand(x.shape[0])]

# we introduce redundancy between the triplet (1, 3, 5) and the first column of
# Y and between (0, 2, 6) and Y
x[:, 1] += y[:, 0]
x[:, 3] += y[:, 0]
x[:, 5] += y[:, 0]
x[:, 0] += y[:, 1]
x[:, 2] += y[:, 1]
x[:, 6] += y[:, 1]

# define the RSI, launch it and inspect the best multiplets
model = RSI(x, y)
hoi = model.fit(minsize=3, maxsize=5)
df = get_nbest_mult(hoi, model=model, minsize=3, maxsize=3, n_best=3)
print(df)

# %%
# This time, as expected, the two most redundant triplets are (1, 3, 5) and
# (0, 2, 6)

###############################################################################
# Simulate univariate and multivariate synergy
# --------------------------------------------
#
# Lets move on to the simulation of synergy that is a bit more subtle. One way
# of simulating synergy is to go the other way of redundancy, meaning we are
# going to add features of `X` inside `Y`. That way, we can only retrieve the
# `Y` variable by knowing the subset of `X`.

# simulate the variable x
x = np.random.rand(200, 7)

# synergy between (0, 3, 5) and 5
y = x[:, 0] + x[:, 3] + x[:, 5]

# define the RSI, launch it and inspect the best multiplets
model = RSI(x, y)
hoi = model.fit(minsize=3, maxsize=5)
df = get_nbest_mult(hoi, model=model, minsize=3, maxsize=3, n_best=3)
print(df)

# %%
# as we can see here, the highest values of higher-order interactions
# (i.e. synergy) is achieved for the multiplet (0, 3, 5). Now we can do the
# same for multivariate synergy

# simulate the variable x
x = np.random.rand(200, 7)

# simulate y and introduce synergy between the subset (0, 3, 5) of x and the
# subset (1, 2, 6)
y = np.c_[x[:, 0] + x[:, 3] + x[:, 5], x[:, 1] + x[:, 2] + x[:, 6]]

# define the RSI, launch it and inspect the best multiplets
model = RSI(x, y)
hoi = model.fit(minsize=3, maxsize=5)
df = get_nbest_mult(hoi, model=model, minsize=3, maxsize=3, n_best=3)
print(df)

###############################################################################
# Combining redundancy and synergy
# --------------------------------
#

# simulate the variable x and y
x = np.random.rand(200, 7)
y = np.random.rand(200, 2)

# synergy between (0, 1, 2) and the first column of y
y[:, 0] = x[:, 0] + x[:, 1] + x[:, 2]

# redundancy between (3, 4, 5) and the second column of x
x[:, 3] += y[:, 1]
x[:, 4] += y[:, 1]
x[:, 5] += y[:, 1]

# define the RSI, launch it and inspect the best multiplets
model = RSI(x, y)
hoi = model.fit(minsize=3, maxsize=5)
df = get_nbest_mult(hoi, model=model, minsize=3, maxsize=3, n_best=3)
print(df)

# plot the result at each order to observe the spreading at orders higher than
# 3
plot_landscape(
    hoi,
    model,
    kind="scatter",
    undersampling=False,
    plt_kwargs=dict(cmap="turbo"),
)
plt.show()
