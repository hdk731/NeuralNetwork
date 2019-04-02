import numpy as np


class Sigmoid(object):
    def __init__(self):
        self.output = None

    # 順伝播 - Forward-Propagation
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    # 誤差逆伝播 - Back-Propagation
    def backward(self, dx):
        return self.output * (1 - self.output)
