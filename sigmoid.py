import numpy as np


class Sigmoid(object):
    def __init__(self):
        self.output = None

    def forward(self, x):
        """
        順伝播 - Forward-Propagation
        :param x: <vector>
        :return: <vector> output
        """

        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, dx):
        """
        誤差逆伝播 - Back-Propagation
        :param dx: <vector> l+1層の微分値
        :return: <vector>
        """

        return self.output * (1 - self.output)
