from hoi.simulation import simulate_hoi_gauss

data = simulate_hoi_gauss(n_samples=1000, triplet_character='synergy')

print(data.shape)
# import the O-information
from hoi.metrics import Oinfo

# define the model
model = Oinfo(data)

# compute hoi for multiplets with a minimum size of 3 and maximum size of 3
# using the Gaussian Copula entropy
hoi = model.fit(minsize=3, maxsize=3, method="gc")

from hoi.plot import plot_landscape
from hoi.utils import get_nbest_mult

# plot the landscape
plot_landscape(hoi, model=model)

# print the summary table
print(get_nbest_mult(hoi, model=model))