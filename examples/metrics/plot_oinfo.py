"""
Metrics example 1
============================================
Lorem ipsum, dolor sit amet consectetur adipisicing elit. Nesciunt neque,
perferendis sed id doloribus at error in ea nulla reprehenderit optio rerum
nostrum harum voluptas, sequi laudantium numquam accusantium consequuntur minus
quisquam autem magnam distinctio. Itaque, fuga quisquam. Dolorum commodi
molestiae soluta, dolorem tenetur magni illum distinctio minus vero tempore
quibusdam totam sed, sapiente aliquam aperiam sunt accusantium quisquam eos
rerum cupiditate fuga debitis inventore. Exercitationem nihil ipsa aliquid,
placeat laborum ut vel odio deserunt adipisci minima blanditiis aperiam nisi
veniam cum labore deleniti reiciendis earum eligendi dignissimos suscipit
beatae ex a! Omnis commodi modi magnam esse molestias odit, non unde eius vel,
aspernatur perferendis voluptatibus nam nihil, aut et. Molestias, quos quia
soluta minus illum amet cupiditate iure a assumenda debitis sint dolor, dolorem
at.
"""
import numpy as np

from hoi.metrics import Oinfo, InfoTopo
from hoi.plot import plot_landscape
from hoi.utils import get_nbest_mult
from hoi.simulation import simulate_hois_gauss

import matplotlib.pyplot as plt
plt.style.use("ggplot")

x = simulate_hois_gauss()

model = InfoTopo(x)
hoi = model.fit(minsize=3, maxsize=None, method="gc")

print(get_nbest_mult(hoi, model))

plot_landscape(hoi, model=model, plt_kwargs=dict(cmap="turbo"))
plt.tight_layout()
plt.show()
