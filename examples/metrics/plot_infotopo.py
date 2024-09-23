"""
Topological Information : conditioning on orders
================================================

The following example highlights a metric known as "Topological Information"
which was introduced by Baudot et al., 2019 :cite:`baudot2019infotopo`. This
metric, founded on disparities in entropies, enables estimation of Higher Order
Interactions while possessing the notable characteristic of being able to
segregate specific orders. Specifically, when simulating redundancies at the
third order utilizing the O-information metric (:class:`hoi.metrics.Oinfo`),
any quadruplets containing the superfluous triplet will yield redundancy.
Conversely, the InfoTopo metric (:class:`hoi.metrics.InfoTopo`) has the
capability of pinpointing exclusively the redundant triplet. You can also check
the original implementation of this estimator :
https://infotopo.readthedocs.io/
"""

# %%
import numpy as np

from hoi.metrics import InfoTopo, Oinfo
from hoi.utils import get_nbest_mult
from hoi.plot import plot_landscape

import matplotlib.pyplot as plt

plt.style.use("ggplot")

###############################################################################
# Simulate redundant and synergistic interactions
# -----------------------------------------------
#
# Let's start by simulating a small network of 7 nodes with 1000 samples each.
# Then we're going to introduce redundancy between the triplet (0, 1, 2) and
# synergy between the quadruplet (3, 4, 5, 6). For further information about
# how to simulate redundant and synergistic interactions, checkout the example
# :ref:`sphx_glr_auto_examples_tutorials_plot_sim_red_syn.py`

# 7 nodes network with 1000 samples each
x = np.random.rand(1000, 7)

# redundant interactions between triplet of nodes (0, 1, 2)
x[:, 1] += x[:, 0]
x[:, 2] += x[:, 0]

# synergistic interactions between quadruplet of nodes (3, 4, 5, 6)
x[:, 3] += x[:, 4] + x[:, 5] + x[:, 6]

###############################################################################
# Spatial spreading : the problem with the O-information
# ------------------------------------------------------
#
# Let's compute the HOI using the O-information

model = Oinfo(x)
hoi = model.fit(minsize=3, method="gc")


# %%
# Now we can plot the landscape. This landscape show the values of HOI for
# different orders. As a reminder, for the O-information, positive values stand
# for redundant interactions while negative values stand for synergistic
# interactions.

plot_landscape(
    hoi,
    model=model,
    kind="scatter",
    plt_kwargs=dict(cmap="Spectral_r"),
    undersampling=False,
)
plt.show()

# %%
# we can also print the multiplets with the highest values of O-information

print(get_nbest_mult(hoi, model=model))

# %%
# As we can see from the landscape and the printed table, the triplet (0, 1, 2)
# with redundant interactions is present in all of the multiplets of higher
# orders (order 4, 5, 6). Same thing holds with the synergistic quadruplets
# (3, 4, 5, 6). In short, the O-information can't isolate both multiplets.

###############################################################################
# Multiplet isolation using the Topological Information
# -----------------------------------------------------
#
# In contrast to the O-information, the Topological Information is based on
# conditional mutual information and conditioned on lower orders. This
# conditioning should, in theory, avoid the spatial spreading. The mathematical
# definition of the Topological Information is given by :
#
# .. math::
#     I_{k}(X_{1}; ...; X_{k}) = \sum_{i=1}^{k} (-1)^{i - 1} \sum_{
#         I\subset[k];card(I)=i} H_{i}(X_{I})

model = InfoTopo(x)
hoi = model.fit(minsize=3, method="gc")


# %%
# Again, we can plot the landscape. This time, as we can see, there's no more
# spatial spreading. There are only two points and, using the printed table
# below, we can see that those two points correspond to our two multiplets
# (0, 1, 2) and (3, 4, 5, 6). However, the Topological Information did not
# correctly inferred the type of interactions as the quadruplet (3, 4, 5, 6)
# is identified as a redundant multiplet despite being synergistic.

plot_landscape(
    hoi,
    model=model,
    kind="scatter",
    plt_kwargs=dict(cmap="Spectral_r"),
    undersampling=False,
)
plt.show()

print(get_nbest_mult(hoi, model=model))
