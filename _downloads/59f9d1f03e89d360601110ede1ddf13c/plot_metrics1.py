"""
Metrics example 1
============================================
Lorem ipsum, dolor sit amet consectetur adipisicing elit.
"""
import numpy as np

from hoi.metrics import Oinfo
from hoi.plot import plot_landscape

import matplotlib.pyplot as plt

plt.style.use("ggplot")

x = np.random.rand(500, 10)
# x[:, 7] = x[:, 0] + x[:, 1]
# x[:, 8] = x[:, 3]
# x[:, 9] = x[:, 3]


model = Oinfo(x)
hoi = model.fit(minsize=3, maxsize=None, method="gcmi")

plot_landscape(hoi, model=model, plt_kwargs=dict(cmap="turbo"))
plt.tight_layout()
plt.show()
