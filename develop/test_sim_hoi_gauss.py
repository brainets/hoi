# %%
from hoi.simulation import simulate_hoi_gauss
from hoi.metrics import Oinfo, GradientOinfo

out = simulate_hoi_gauss(
    n_samples=100, target=True, triplet_character="redundancy"
)

model = GradientOinfo(out[0], out[1], verbose=False)
hoi = model.fit(minsize=3, maxsize=3, method="gauss")

print(hoi)