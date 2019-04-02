import numpy as np
import nn
from sigmoid import Sigmoid
from MSE import MSE

fc1 = nn.Network(10, 5, 0.1)
sig1 = Sigmoid()
fc2 = nn.Network(5, 2, 0.1)
sig2 = Sigmoid()
mse = MSE
# 学習データ生成
x = np.random.randn(10)
# 教師データ生成
t = np.random.randn(2)
for i in range(100):
    print("0")
    out = sig2.forward(fc2.forward(sig1.forward(fc1.forward(x))))
    print("1")
    print("out:")
    print(out.reshape(-1))
    print("t:")
    print(t.reshape(-1))
    aaa = out.reshape(-1) + out.reshape(-1)
    print(aaa)
    loss = mse.forward(out, t)
    print("2")
    grad = mse.backward(out, t)
    print("3")
    print(loss, grad)
    fc1.backward(sig1.backward(fc2.backward(sig2.backward(grad))))
    fc1.update()
    fc2.update()
