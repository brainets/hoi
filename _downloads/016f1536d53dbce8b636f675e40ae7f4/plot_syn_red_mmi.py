"""
Redundancy and Synergy MMI
==========================

This example illustrates how to use and interpret synergy and redundancy
computed using the Minimum Mutual Information (MMI) approach to 
approximate the redundancy.
"""

# %%
import numpy as np

from hoi.metrics import SynergyMMI, RedundancyMMI
from hoi.utils import get_nbest_mult
from hoi.plot import plot_landscape

import matplotlib.pyplot as plt

plt.style.use("ggplot")


###############################################################################
# Definition
# ----------
#
# :term:`Synergy` and :term:`redundancy` measures directly, respectively the ammount
# of synergy and redundancy carried by a group of variable :math:`S`
# about a target variable :math:`Y`. Synergy is defined as follow :

# %%
# .. math::
#     Syn(S; Y) \equiv I(S; Y) - \max_{x_{i}\in S} I(S_{-i}; Y)
#
# with : :math:`S = x_{1}, ..., x_{n}` and
# :math:`S_{-i} = x_{1}, ..., x_{i-1}, x_{1+1}, ..., x_{n}`
#
# Positive values of Synergy stand for the presence of an higher information
# about :math:`Y` when considering all the variables in :math:`S` with
# respect to when considering only a subgroup of :math:`n-1`.
#
# Redundancy, in the approximation based on the Minimum Mutual Information
# (MMI) framework, is computed as follow :

# %%
# .. math::
#     Red(S; Y) \equiv \min_{x_{i}\in S} I(X_{i}; Y)

###############################################################################
# Simulate univariate redundancy
# ------------------------------
#
# A very simple way to simulate redundancy is to observe that if a triplet of
# variables :math:`X_{1}, X_{2}, X_{3}` receive a copy of a variable :math:`Y`,
# we will observe redundancy between :math:`X_{1}, X_{2}, X_{3}` and :math:`Y`.
# For further information about how to simulate redundant and synergistic
# interactions, checkout the example
# :ref:`sphx_glr_auto_examples_tutorials_plot_sim_red_syn.py`.

# lets start by simulating a variable x with 200 samples and 7 features
x = np.random.rand(200, 7)

# now we can also generate a univariate random variable `Y`
y = np.random.rand(x.shape[0])

# we now send the variable y in the column (1, 3, 5) of `X`
x[:, 1] += y
x[:, 3] += y
x[:, 5] += y

# define the RSI model and launch it
model = RedundancyMMI(x, y)
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
# `Y` and between (0, 2, 6) and the second column of `Y`.
x[:, 1] += y[:, 0]
x[:, 3] += y[:, 0]
x[:, 5] += y[:, 0]
x[:, 0] += y[:, 1]
x[:, 2] += y[:, 1]
x[:, 6] += y[:, 1]

# define the Redundancy, launch it and inspect the best multiplets
model = RedundancyMMI(x, y)
hoi = model.fit(minsize=3, maxsize=5)
df = get_nbest_mult(hoi, model=model, minsize=3, maxsize=3, n_best=5)
print(df)

# %%
# It is important to notice that in this case the redundancy it is not able to
# find only the two multiplets in which we generate redundancy, but instead
# all the possible combination of the six variables in which a part of `Y`
# was copied are resulting redundant. This follows directly by the definition
# of redundancy with the MMI approximation. It is important to remember this
# limitation when leveraging its results and eventually
# consider a double check with other metrics, as RSI or O-information gradient.

###############################################################################
# Simulate univariate and multivariate synergy
# --------------------------------------------
#
# Lets move on to the simulation of synergy that is a bit more subtle. One way
# of simulating synergy is to go the other way of redundancy, meaning we are
# going to add features of x inside `Y`. That way, we can only retrieve the `Y`
# variable by knowing the subset of `X`.

# simulate the variable x
x = np.random.rand(200, 7)

# synergy between (0, 3, 5) and 5
y = x[:, 0] + x[:, 3] + x[:, 5]

# define the Synergy, launch it and inspect the best multiplets
model = SynergyMMI(x, y)
hoi = model.fit(minsize=3, maxsize=5)
df = get_nbest_mult(hoi, model=model, minsize=3, maxsize=3, n_best=3)
print(df)

# %%
# as we can see here, the highest values of higher-order interactions
# (i.e. synergy) is achieved for the multiplet (0, 3, 5). Now we can do the
# same for multivariate synergy.

# simulate the variable `X`
x = np.random.rand(200, 7)

# simulate y and introduce synergy between the subset (0, 3, 5) of `X` and the
# subset (1, 2, 6)
y = np.c_[x[:, 0] + x[:, 3] + x[:, 5], x[:, 1] + x[:, 2] + x[:, 6]]

# define the Synergy, launch it and inspect the best multiplets
model = SynergyMMI(x, y)
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

# synergy between (0, 1, 2) and the first column of `Y`
y[:, 0] = x[:, 0] + x[:, 1] + x[:, 2]

# redundancy between (3, 4, 5) and the second column of `X`
x[:, 3] = y[:, 1]
x[:, 4] = y[:, 1]
x[:, 5] = y[:, 1]

# define the Synergy, launch it and inspect the best multiplets
model_syn = SynergyMMI(x, y)
hoi_syn = model_syn.fit(minsize=3, maxsize=5)
df = get_nbest_mult(hoi_syn, model=model, minsize=3, maxsize=3, n_best=3)
print(" ")
print("Synergy results")
print(" ")
print(df)

# define the Synergy, launch it and inspect the best multiplets
model_red = RedundancyMMI(x, y)
hoi_red = model_red.fit(minsize=3, maxsize=5)
df = get_nbest_mult(hoi_red, model=model, minsize=3, maxsize=3, n_best=3)
print(" ")
print("Redundancy results")
print(" ")
print(df)

# %%
# Plotting redundancy
plot_landscape(
    hoi_red,
    model_red,
    kind="scatter",
    undersampling=False,
    plt_kwargs=dict(cmap="turbo"),
)

plt.show()

# %%
# Plotting synergy

plot_landscape(
    hoi_syn,
    model_syn,
    kind="scatter",
    undersampling=False,
    plt_kwargs=dict(cmap="turbo"),
)

plt.show()