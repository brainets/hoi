"""
Integrated Information Decomposition
====================================

This example illustrates how to use and interpret synergy and redundancy as
defined in the Integrated Information Decomposition framework
"""

import numpy as np

from hoi.metrics import SynergyphiID, RedundancyphiID
from hoi.utils import get_nbest_mult

import matplotlib.pyplot as plt

plt.style.use("ggplot")


###############################################################################
# Definition
# ----------
#
# The synergy as defined in the Integrated Information decomposition framework
# is a pairwise measure of the synergistic information that two variables
# carry about their own future. For a couple of variables,
# :math:`X` and  :math:`Y`, when using the minimum mutual information (MMI)
# approximation for the redundancy, it is defined in the following way:

# %%
# .. math::
#     Syn(X,Y) =  I(X_{t-\tau},Y_{t-\tau};X_{t},Y_t) -
#                 max \{ I(X_{t-\tau};X_t,Y_t),I(Y_{t-\tau};X_t,Y_t) \}
#
# Synergy is a positive defined measures that relates to emergent properties of
# the couple of variables under study. It measures how much we can predict the
# future state of the couple of variables when considering them as a whole with
# respect to when considering them separately.
# Redundancy instead, following the MMI framework approximation is computed in
# the following way:

# %%
# .. math::
#     Red(X,Y) =  min (I(X_{t-\tau}; Y_{t}), I(X_{t-\tau}; X_t),
#                      I(Y_{t-\tau}; X_t), I(Y_{t-\tau}; Y_t))
#
# Redundancy relates to the amount of information the two variables share
# about their own future. A high presence of redundancy can be associated with
# robustness, while a stronger presence of synergy to emergence.

###############################################################################
# Simulate synergy
# ------------------
#
# A very simple way to simulate synergy is by to start with two independent
# variables and inflate autocorrelation by
# variables :math:`X_{1}, X_{2}, X_{3}` receive a copy of a variable :math:`Y`,
# we will observe redundancy between :math:`X_{1}, X_{2}, X_{3}` and :math:`Y`.

# lets start by simulating a variable x with 200 samples and 7 features
x = np.random.rand(200, 7)

# now to create synergy between the two first features, we do the following:
# to create interdependencies between past and future we use a uniform
# kernel in the following way

for i in range(190):
    x[i, 0] = np.sum(x[i : i + 20, 1]) + 0.2 * np.sum(x[i : i + 20, 0])
    x[i, 1] = np.sum(x[i : i + 20, 0]) + 0.2 * np.sum(x[i : i + 20, 1])

# define the SynergyphiID model and launch it
model = SynergyphiID(x)
hoi = model.fit(minsize=2, maxsize=2)

# now we can take a look at the multiplets with the highest and lowest values
# of synergy. We will only select the multiplets of size 2 here
df = get_nbest_mult(hoi, model=model, minsize=2, maxsize=2, n_best=3)
print(df)

# %%
# as you see from the printed table, the couple of variables with the highest
# synergy is [0,1]

###############################################################################
# Simulate redundancy
# -------------------


# simulate x again
x = np.random.rand(200, 7)

# now to create synergy between the two first features, we do the following:
# to create interdependencies between past and future we use a uniform
# kernel in the following way

for i in range(190):
    x[i, 0] = np.sum(x[i : i + 20, 1])

# Redundancy emerges when the two variables carry the same information about
# Their future. This can be achieved by copy operation between the two
# variables plus some noise.

x[:, 1] = x[:, 0] + np.random.rand(200) * 0.05

# define the redundancyphiID, launch it and inspect the best multiplets
model = RedundancyphiID(x)
hoi = model.fit(minsize=2, maxsize=2)
df = get_nbest_mult(hoi, model=model, minsize=2, maxsize=2, n_best=3)
print(df)

# %%
# This time, as expected, the two most redundant couple of variable is [0,1]


###############################################################################
# Combining redundancy and synergy
# --------------------------------
#

# simulate the variable x
n_features = 7
x = np.random.rand(200, n_features)

# synergy between (0, 1)
for i in range(190):
    x[i, 0] = np.sum(x[i : i + 20, 1]) + 0.1 * np.sum(x[i : i + 20, 0])
    x[i, 1] = np.sum(x[i : i + 20, 0]) + 0.1 * np.sum(x[i : i + 20, 1])

# redundancy between (0, 2), this will inject also synergy between
# variables 1 and 2

x[:, 2] = x[:, 0] + np.random.rand(200) * 0.05

# %%
# define the Synergy_phiID, launch it and inspect the best couples of
# variables

model = SynergyphiID(x)
syn_results = model.fit(minsize=2, maxsize=2)
df = get_nbest_mult(hoi, model=model, minsize=2, maxsize=2, n_best=3)
print(df)

# store the results in a matrix

matrix_to_plot = np.zeros((n_features, n_features))
for i, tup in enumerate(model.get_combinations(minsize=2, maxsize=2)[0]):
    m, g = tup
    matrix_to_plot[m, g] = syn_results[i][0]

# plot the results matrix simmetrized

plt.imshow(
    matrix_to_plot + matrix_to_plot.T, aspect="auto", interpolation="none"
)
plt.colorbar()
plt.show()

# %%
# define the Synergy_phiID, launch it and inspect the best couples of
# variables

model = RedundancyphiID(x)
red_results = model.fit(minsize=2, maxsize=2)
df = get_nbest_mult(red_results, model=model, minsize=2, maxsize=2, n_best=3)
print(df)

# store the results in a matrix

matrix_to_plot = np.zeros((n_features, n_features))
for i, tup in enumerate(model.get_combinations(minsize=2, maxsize=2)[0]):
    m, g = tup
    matrix_to_plot[m, g] = red_results[i][0]

# plot the results matrix

plt.imshow(
    matrix_to_plot + matrix_to_plot.T, aspect="auto", interpolation="none"
)
plt.colorbar()
plt.show()
