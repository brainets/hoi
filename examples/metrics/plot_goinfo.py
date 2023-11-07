import numpy as np
from hoi.utils import get_nbest_mult
from hoi.metrics import GradientOinfo

np.random.seed(0)

x = np.random.rand(200, 7)
y_red = np.random.rand(x.shape[0])

# redundancy: (1, 2, 6) + (7, 8)
x[:, 1] += y_red
x[:, 2] += y_red
x[:, 6] += y_red
# synergy:    (0, 3, 5) + (7, 8)
y_syn = x[:, 0] + x[:, 3] + x[:, 5]
# bivariate target
y = np.c_[y_red, y_syn]

model = GradientOinfo(x, y=y)
hoi = model.fit(minsize=2, maxsize=None, method="gcmi")

print(get_nbest_mult(hoi, model=model, minsize=3, maxsize=3, n_best=3))