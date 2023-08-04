"""
Metrics example 1
============================================
Lorem ipsum, dolor sit amet consectetur adipisicing elit. Nesciunt neque, perferendis sed id doloribus at error in ea nulla reprehenderit optio rerum nostrum harum voluptas, sequi laudantium numquam accusantium consequuntur minus quisquam autem magnam distinctio. Itaque, fuga quisquam. Dolorum commodi molestiae soluta, dolorem tenetur magni illum distinctio minus vero tempore quibusdam totam sed, sapiente aliquam aperiam sunt accusantium quisquam eos rerum cupiditate fuga debitis inventore. Exercitationem nihil ipsa aliquid, placeat laborum ut vel odio deserunt adipisci minima blanditiis aperiam nisi veniam cum labore deleniti reiciendis earum eligendi dignissimos suscipit beatae ex a! Omnis commodi modi magnam esse molestias odit, non unde eius vel, aspernatur perferendis voluptatibus nam nihil, aut et. Molestias, quos quia soluta minus illum amet cupiditate iure a assumenda debitis sint dolor, dolorem at. Tempora, dolorum maiores! Explicabo molestias nostrum nihil non laboriosam accusantium aut maxime. Tenetur similique natus quos accusamus, dicta iure? Exercitationem id aut laborum itaque consequatur delectus molestias labore sequi. Libero distinctio laboriosam debitis possimus beatae nisi quae autem in voluptatibus labore vero error omnis neque, quisquam culpa odio! Tempore neque ipsa voluptatibus alias laudantium id voluptates reiciendis dolorem iusto cumque tenetur illum aliquid maiores totam sint rem repellat, repellendus molestias deserunt deleniti odit mollitia?
"""
import numpy as np

from hoi.metrics import Oinfo
from hoi.plot import plot_landscape

import matplotlib.pyplot as plt

plt.style.use('ggplot')

x = np.random.rand(500, 10)
# x[:, 7] = x[:, 0] + x[:, 1]
# x[:, 8] = x[:, 3]
# x[:, 9] = x[:, 3]


model = Oinfo(x)
hoi = model.fit(minsize=3, maxsize=None, method='gcmi')

plot_landscape(hoi, model=model,
               plt_kwargs=dict(cmap='turbo'))
plt.tight_layout()
plt.show()
