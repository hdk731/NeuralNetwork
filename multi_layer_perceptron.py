from nn import nn
from sigmoid import sigmoid
fc1 = nn(10, 5, 0.1)
sig1 = sigmoid()
fc2 = nn(5, 2, 0.1)
sig2 = sigmoid()
x = np.random.randn(10, 1)
sig2.forward(fc2.forward(sig1.forward(fc1.forward(x))))
grad = np.random.randn(1,2)
fc1.backward(sig1.backward(fc2.backward(sig2.backward(grad))))
fc1.update()
fc2.update()