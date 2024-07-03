import numpy as np
from hoi.metrics import Oinfo

def method(x):
    return np.mean(x, axis=1)

x = np.random.rand(200, 3)

model = Oinfo(x)
hoi = model.fit(method=method)
print(hoi)
