import numpy as np
import Network as nn
from sigmoid import Sigmoid
from MeanSquaredError import MeanSquaredError as MSE


fc1 = nn.Network(10, 5, 0.1)
sig1 = Sigmoid()
fc2 = nn.Network(5, 2, 0.1)
sig2 = Sigmoid()
mse = MSE()
# 学習データ生成
x = np.random.randn(10)
# 教師データ生成
t = np.random.randn(2)

for i in range(100):
    out = sig2.forward(fc2.forward(sig1.forward(fc1.forward(x))))
    loss = mse.forward(out, t)
    grad = mse.backward(out, t)
    print(loss)
    fc1.backward(sig1.backward(fc2.backward(sig2.backward(grad))))
    fc1.update()
    fc2.update()
